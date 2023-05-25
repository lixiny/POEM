import os
from collections import OrderedDict
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from manotorch.manolayer import ManoLayer
from lib.external.metro.hrnet.config import config as hrnet_config
from lib.external.metro.hrnet.config import \
    update_config as hrnet_update_config
from lib.external.metro.hrnet.hrnet import get_cls_net
from lib.metrics.basic_metric import LossMetric
from lib.metrics.pa_eval import PAEval
from lib.metrics.mean_epe import MeanEPE
from lib.models.backbones import create_backbone
from lib.models.layers.mano_wrapper import kpId2vertices
from lib.models.model_abstraction import ModuleAbstract
from lib.utils.builder import MODEL
from lib.utils.logger import logger
from lib.utils.misc import CONST, enable_lower_param, param_size
from lib.utils.transform import bchw_2_bhwc, denormalize
from lib.viztools.draw import draw_batch_joint_images, plot_hand
from transformers.models.bert.modeling_bert import BertConfig

from .base_model import MeshSampler
from .base_model import METRO_Hand_Network as METRO_Network
from .base_model import METROBlock


@MODEL.register_module()
class METRO(nn.Module, ModuleAbstract):

    def __init__(self, cfg):
        super().__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL

        input_feat_dim = cfg.INPUT_FEAT_DIM
        hidden_feat_dim = cfg.HIDDEN_FEAT_DIM
        output_feat_dim = input_feat_dim[1:] + [3]

        self.num_hidden_layers = cfg.NUM_HIDDEN_LAYERS
        self.num_attention_heads = cfg.NUM_ATTENTION_HEADS

        self.VLOSS_W_SUB = cfg.VLOSS_W_SUB
        self.VLOSS_W_FULL = cfg.VLOSS_W_FULL

        self.JOINTS_LOSS_WEIGHT = cfg.JOINTS_LOSS_WEIGHT
        self.VERTICES_LOSS_WEIGHT = cfg.VERTICES_LOSS_WEIGHT

        self.PA_metric = PAEval(cfg.METRIC)
        self.mepe3d_metric = MeanEPE(cfg, "joints_3d_rel")
        self.loss_metric = LossMetric(cfg)

        # * load metro block
        self.trans_encoder = []
        # init three transformer-encoder blocks in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METROBlock

            self.dropout = cfg.DROP_OUT

            config = config_class.from_pretrained("lib/external/metro/bert_cfg.json")
            config.output_attentions = False
            config.hidden_dropout_prob = self.dropout
            config.img_feature_dim = input_feat_dim[i]
            config.output_feature_dim = output_feat_dim[i]
            self.hidden_size = hidden_feat_dim[i]
            self.intermediate_size = self.hidden_size * 4

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for idx, param in enumerate(update_params):
                arg_param = getattr(self, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            logger.info("Init model from scratch.")
            self.trans_encoder.append(model)

        # * load metro backbone
        if cfg.BACKBONE == "hrnet":
            hrnet_yaml = "lib/external/metro/hrnet/config/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
            hrnet_checkpoint = "checkpoints/hrnetv2_w40_imagenet_pretrained.pth"
            hrnet_update_config(hrnet_config, hrnet_yaml)
            self.backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info("=> loading hrnet-v2-w40 model")
        elif cfg.BACKBONE == "hrnet-w64":
            hrnet_yaml = "lib/external/metro/hrnet/config/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml"
            hrnet_checkpoint = "checkpoints/hrnetv2_w64_imagenet_pretrained.pth"
            hrnet_update_config(hrnet_config, hrnet_yaml)
            self.backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info("=> loading hrnet-v2-w64 model")

        self.trans_encoder = torch.nn.Sequential(*self.trans_encoder)
        total_params = param_size(self.trans_encoder)
        logger.info("Transformers total parameters: {}M".format(total_params))
        backbone_total_params = param_size(self.backbone)
        logger.info("Backbone total parameters: {}M".format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        self.metro_network = METRO_Network(config, self.backbone, self.trans_encoder)

        self.mesh_sampler = MeshSampler()
        self.mano_layer = ManoLayer(joint_rot_mode="axisang",
                                    use_pca=False,
                                    mano_assets_root="assets/mano_v1_2",
                                    center_idx=cfg.DATA_PRESET.CENTER_IDX,
                                    flat_hand_mean=True)

        # * criterion
        self.criterion_2d_keypoints = torch.nn.MSELoss(reduction="none")
        self.criterion_keypoints = torch.nn.MSELoss(reduction="none")
        self.criterion_vertices = torch.nn.L1Loss()

        # if args.resume_checkpoint != None and args.resume_checkpoint != "None":
        #     # for fine-tuning or resume training or inference, load weights from checkpoint
        #     logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        #     cpu_device = torch.device("cpu")
        #     state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        #     _metro_network.load_state_dict(state_dict, strict=False)
        #     del state_dict

        if cfg.PRETRAINED:
            logger.warning(f"{self.name} re-initalized by {cfg.PRETRAINED}")
        self.init_weights(pretrained=cfg.PRETRAINED)
        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def forward(self, inputs, step_idx, mode="train", **kwargs):
        if mode == "train":
            return self.training_step(inputs, step_idx, **kwargs)
        elif mode == "val":
            return self.validation_step(inputs, step_idx, **kwargs)
        elif mode == "test":
            return self.testing_step(inputs, step_idx, **kwargs)
        elif mode == "inference":
            return self.inference_step(inputs, step_idx, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode}")

    def validation_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch)
        img = batch["image"]
        self.PA_metric.feed(preds["pred_3d_joints"], batch["target_joints_3d_rel"], preds["pred_vertices"],
                            batch["target_verts_3d_rel"])
        self.summary.add_scalar("pa_jt3d_val", self.PA_metric.get_result(), step_idx)
        if step_idx % (self.train_log_interval * 10) == 0:  # viz every 10 * interval batches
            img_array = draw_batch_joint_images(preds["pred_2d_joints"] * img.shape[-1], batch["target_joints_2d"], img,
                                                step_idx)
            self.summary.add_images("img/joints2d_val", img_array, step_idx, dataformats="NHWC")
        return preds, {}

    def on_val_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-test"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([self.PA_metric], epoch_idx, comment=comment)
        self.loss_metric.reset()
        self.PA_metric.reset()
        return

    def training_step(self, batch, step_idx, **kwargs):
        batch_size = batch["image"].shape[0]
        img = batch["image"]
        preds = self._forward_impl(batch)
        loss, loss_dict = self.compute_loss(preds, batch)
        self.loss_metric.feed(loss_dict, batch_size)
        self.mepe3d_metric.feed(preds["pred_3d_joints"], gt_kp=batch["target_joints_3d_rel"])

        if step_idx % self.train_log_interval == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(f"{k}", v.item(), step_idx)
            self.summary.add_scalar("jt3d_mepe", self.mepe3d_metric.get_result(), step_idx)

            if step_idx % (self.train_log_interval * 10) == 0:  # viz every 10 * interval batches
                img_array = draw_batch_joint_images(preds["pred_2d_joints"] * img.shape[-1], batch["target_joints_2d"],
                                                    img, step_idx)
                self.summary.add_images("img/joints2d", img_array, step_idx, dataformats="NHWC")
        return preds, loss_dict

    def on_train_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)

        recorder.record_metric([self.PA_metric], epoch_idx, comment=comment)
        recorder.record_metric([self.mepe3d_metric], epoch_idx, comment=comment)

        self.loss_metric.reset()
        self.PA_metric.reset()
        self.mepe3d_metric.reset()

    def format_metric(self, mode="train"):
        if mode == "train":
            return f"{self.loss_metric.get_loss('loss'):.4f}"
        else:
            metric_toshow = [self.PA_metric, self.mepe3d_metric]
            return " | ".join([str(me) for me in metric_toshow])

    @staticmethod
    def orthographic_projection(X, camera):
        """Perform orthographic projection of 3D points X using the camera parameters
        Args:
            X: size = [B, N, 3]
            camera: size = [B, 3]
        Returns:
            Projected 2D points -- size = [B, N, 2]
        """
        camera = camera.view(-1, 1, 3)
        X_trans = X[:, :, :2] + camera[:, :, 1:]
        shape = X_trans.shape
        X_2d = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
        return X_2d

    @staticmethod
    def keypoint_2d_loss(criterion_keypoints, pred_keypoints_2d, gt_keypoints_2d, has_pose_2d):
        """
        Compute 2D reprojection loss if 2D keypoint annotations are available.
        The confidence is binary and indicates whether the keypoints exist or not.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    @staticmethod
    def keypoint_3d_loss(criterion_keypoints, pred_keypoints_3d, gt_keypoints_3d, has_pose_3d):
        """
        Compute 3D keypoint loss if 3D keypoint annotations are available.
        """
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        gt_keypoints_3d = gt_keypoints_3d[has_pose_3d == 1]
        conf = conf[has_pose_3d == 1]
        pred_keypoints_3d = pred_keypoints_3d[has_pose_3d == 1]
        if len(gt_keypoints_3d) > 0:
            gt_root = gt_keypoints_3d[:, 0, :]
            gt_keypoints_3d = gt_keypoints_3d - gt_root[:, None, :]
            pred_root = pred_keypoints_3d[:, 0, :]
            pred_keypoints_3d = pred_keypoints_3d - pred_root[:, None, :]
            return (conf * criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
        else:
            return torch.FloatTensor(1).fill_(0.0).cuda()

    @staticmethod
    def vertices_loss(criterion_vertices, pred_vertices, gt_vertices, has_smpl):
        """
        Compute per-vertex loss if vertex annotations are available.
        """
        pred_vertices_with_shape = pred_vertices[has_smpl == 1]
        gt_vertices_with_shape = gt_vertices[has_smpl == 1]
        if len(gt_vertices_with_shape) > 0:
            return criterion_vertices(pred_vertices_with_shape, gt_vertices_with_shape)
        else:
            return torch.FloatTensor(1).fill_(0.0).cuda()

    def _forward_impl(self, batch):

        image = batch["image"]

        mjm_mask = batch["mjm_mask"]
        mvm_mask = batch["mvm_mask"]

        # prepare masks for mask vertex/joint modeling
        mjm_mask_ = mjm_mask.expand(-1, -1, 2051)
        mvm_mask_ = mvm_mask.expand(-1, -1, 2051)
        meta_masks = torch.cat([mjm_mask_, mvm_mask_], dim=1)

        # forward-pass
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices = self.metro_network(image,
                                                                                           self.mano_layer,
                                                                                           self.mesh_sampler,
                                                                                           meta_masks=meta_masks,
                                                                                           is_train=True)

        # obtain 3d joints, which are regressed from the full mesh
        # pred_3d_joints_from_mesh = mano_model.get_3d_joints(pred_vertices)
        pred_3d_joints_from_mesh = torch.matmul(self.mano_layer.th_J_regressor, pred_vertices)
        tipsId = [v[0] for k, v in kpId2vertices.items()]
        tips = pred_vertices[:, tipsId]
        pred_3d_joints_from_mesh = torch.cat([pred_3d_joints_from_mesh, tips], dim=1)
        # Reorder joints to match OpenPose definition
        pred_3d_joints_from_mesh = pred_3d_joints_from_mesh[:, [
            0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20
        ]]

        # obtain 2d joints, which are projected from 3d joints of smpl mesh
        pred_2d_joints_from_mesh = self.orthographic_projection(pred_3d_joints_from_mesh.contiguous(),
                                                                pred_camera.contiguous())
        pred_2d_joints = self.orthographic_projection(pred_3d_joints.contiguous(), pred_camera.contiguous())

        res = {
            "pred_camera": pred_camera,
            "pred_3d_joints": pred_3d_joints,
            "pred_vertices_sub": pred_vertices_sub,
            "pred_vertices": pred_vertices,
            "pred_3d_joints_from_mesh": pred_3d_joints_from_mesh,
            "pred_2d_joints_from_mesh": pred_2d_joints_from_mesh,
            "pred_2d_joints": pred_2d_joints,
        }
        return res

    def compute_loss(self, preds, gt):

        batch_size = gt["image"].shape[0]

        pred_3d_joints = preds["pred_3d_joints"]
        pred_2d_joints = preds["pred_2d_joints"]
        pred_2d_joints_from_mesh = preds["pred_2d_joints_from_mesh"]
        pred_3d_joints_from_mesh = preds["pred_3d_joints_from_mesh"]
        pred_vertices = preds["pred_vertices"]
        pred_vertices_sub = preds["pred_vertices_sub"]

        has_mesh = torch.ones(batch_size, device=pred_3d_joints.device)

        gt_2d_joints = gt["target_joints_2d"] / gt["image"].shape[-1]
        gt_vertices = gt["target_verts_3d"]
        gt_3d_joints = gt["target_joints_3d"]

        gt_vertices_sub = self.mesh_sampler.downsample(gt_vertices)
        gt_3d_root = gt["target_root_joint"]
        gt_vertices = gt_vertices - gt_3d_root[:, None, :]
        gt_vertices_sub = gt_vertices_sub - gt_3d_root[:, None, :]
        gt_3d_joints = gt_3d_joints - gt_3d_root[:, None, :]

        gt_3d_joints_with_tag = torch.ones((batch_size, gt_3d_joints.shape[1], 4)).cuda()
        gt_3d_joints_with_tag[:, :, :3] = gt_3d_joints

        gt_2d_joints_with_tag = torch.ones((batch_size, gt_2d_joints.shape[1], 3)).cuda()
        gt_2d_joints_with_tag[:, :, :2] = gt_2d_joints

        # compute 3d joint loss  (where the joints are directly output from transformer)
        loss_3d_joints = self.keypoint_3d_loss(self.criterion_keypoints, pred_3d_joints, gt_3d_joints_with_tag,
                                               has_mesh)

        # compute 3d vertex loss
        loss_vertices = self.VLOSS_W_SUB * self.vertices_loss(
            self.criterion_vertices, pred_vertices_sub, gt_vertices_sub, has_mesh
        ) + self.VLOSS_W_FULL * self.vertices_loss(self.criterion_vertices, pred_vertices, gt_vertices, has_mesh)

        # compute 3d joint loss (where the joints are regressed from full mesh)
        loss_reg_3d_joints = self.keypoint_3d_loss(self.criterion_keypoints, pred_3d_joints_from_mesh,
                                                   gt_3d_joints_with_tag, has_mesh)
        # compute 2d joint loss
        loss_2d_joints = self.keypoint_2d_loss(
            self.criterion_2d_keypoints, pred_2d_joints, gt_2d_joints_with_tag, has_mesh) + self.keypoint_2d_loss(
                self.criterion_2d_keypoints, pred_2d_joints_from_mesh, gt_2d_joints_with_tag, has_mesh)

        loss_3d_joints = loss_3d_joints + loss_reg_3d_joints

        # we empirically use hyperparameters to balance difference losses
        loss = (self.JOINTS_LOSS_WEIGHT * loss_3d_joints + self.VERTICES_LOSS_WEIGHT * loss_vertices +
                self.JOINTS_LOSS_WEIGHT * loss_2d_joints)

        loss_dict = {}

        loss_dict["loss_3d_joints"] = loss_3d_joints.mean().detach()
        loss_dict["loss_vertices"] = loss_vertices.mean().detach()
        loss_dict["loss_2d_joints"] = loss_2d_joints.mean().detach()
        loss_dict['loss'] = loss

        return loss, loss_dict

    def testing_step(self, batch, step_idx, **kwargs):
        return self.validation_step(batch, step_idx, **kwargs)

    def on_test_finished(self, recorder, epoch_idx):
        return self.on_val_finished(recorder, epoch_idx)

    def inference_step(self, batch, step_idx, **kwargs):
        raise NotImplementedError()

    def init_weights(self, pretrained=None):
        if pretrained is None or pretrained == "":
            logger.warning(f"=> Train {type(self).__name__} weights from the scratch.")
        elif os.path.isfile(pretrained):
            from collections import OrderedDict

            # pretrained_state_dict = torch.load(pretrained)
            logger.info(f"=> Loading {self.name} pretrained model from: {pretrained}")
            # self.load_state_dict(pretrained_state_dict, strict=False)
            checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
            if isinstance(checkpoint, OrderedDict):
                state_dict = checkpoint
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict_old = checkpoint["state_dict"]
                state_dict = OrderedDict()
                # delete 'module.' because it is saved from DataParallel module
                for key in state_dict_old.keys():
                    if key.startswith("module."):
                        # state_dict[key[7:]] = state_dict[key]
                        # state_dict.pop(key)
                        state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                    else:
                        state_dict[key] = state_dict_old[key]
            else:
                logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
                raise RuntimeError()
            self.load_state_dict(state_dict, strict=True)
            logger.info(f"=> Loading SUCCEEDED")
        else:
            logger.error(f"=> No {type(self).__name__} checkpoints file found in {pretrained}")
            raise FileNotFoundError()
