import math
import numpy as np
import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer

from ..metrics.basic_metric import LossMetric
from ..metrics.mean_epe import MeanEPE
from ..metrics.pa_eval import PAEval
from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import param_size
from ..utils.net_utils import init_weights
from ..utils.recorder import Recorder
from ..utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, batch_persp_project, mano_to_openpose)
from ..viztools.draw import (draw_batch_joint_images, draw_batch_mesh_images, draw_batch_verts_images)
from .backbones import build_backbone
from .heads import build_head
from .model_abstraction import ModuleAbstract


@MODEL.register_module()
class PETRMultiView(nn.Module, ModuleAbstract):

    def __init__(self, cfg):
        super(PETRMultiView, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_cfg = cfg.TRAIN
        self.data_preset_cfg = cfg.DATA_PRESET
        self.num_joints = cfg.DATA_PRESET.NUM_JOINTS
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX

        self.pred_joints_from_mesh = cfg.get("PRED_JOINTS_FROM_MESH", False)
        # self.loss_joint_3d_use_mesh = cfg.LOSS.get("USE_MESH_REGRESSION_JOINT", True)

        self.img_backbone = build_backbone(cfg.BACKBONE, data_preset=self.data_preset_cfg)
        self.head = build_head(cfg.HEAD, data_preset=self.data_preset_cfg)
        self.num_preds = self.head.num_preds
        self.mano_layer = ManoLayer(joint_rot_mode="axisang",
                                    use_pca=False,
                                    mano_assets_root="assets/mano_v1_2",
                                    center_idx=cfg.DATA_PRESET.CENTER_IDX,
                                    flat_hand_mean=True)

        self.face = self.mano_layer.th_faces

        self.joints_loss_type = cfg.LOSS.get("JOINTS_LOSS_TYPE", "l2")
        self.verts_loss_type = cfg.LOSS.get("VERTICES_LOSS_TYPE", "l1")
        self.joints_weight = cfg.LOSS.JOINTS_LOSS_WEIGHT
        self.vertices_weight = cfg.LOSS.VERTICES_LOSS_WEIGHT
        self.joints_2d_weight = cfg.LOSS.get("JOINTS_2D_LOSS_WEIGHT", 0.0)
        self.vertices_2d_weight = cfg.LOSS.get("VERTICES_2D_LOSS_WEIGHT", 0.0)
        self.edge_weight = cfg.LOSS.get("EDGE_LOSS_WEIGHT", 0.0)

        if self.joints_loss_type == "l2":
            self.criterion_joints = torch.nn.MSELoss()
        else:
            self.criterion_joints = torch.nn.L1Loss()

        if self.verts_loss_type == "l2":
            self.criterion_vertices = torch.nn.MSELoss()
        else:
            self.criterion_vertices = torch.nn.L1Loss()

        self.loss_metric = LossMetric(cfg)
        self.PA = PAEval(cfg, mesh_score=True)
        self.MPJPE_3D = MeanEPE(cfg, "joints_3d")
        self.MPVPE_3D = MeanEPE(cfg, "vertices_3d")
        self.MPJPE_3D_REL = MeanEPE(cfg, "joints_3d_rel")
        self.MPVPE_3D_REL = MeanEPE(cfg, "vertices_3d_rel")

        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        init_weights(self, pretrained=cfg.PRETRAINED)
        logger.info(f"{self.name} has {param_size(self)}M parameters")
        logger.info(f"{self.name} loss type: joint {self.joints_loss_type} verts {self.verts_loss_type}")

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def extract_img_feat(self, img):
        B = img.size(0)
        if img.dim() == 5:
            if img.size(0) == 1 and img.size(1) != 1:  # (1, N, C, H, W)
                img = img.squeeze(0)  # (N, C, H, W)
            else:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(image=img)
        global_feat = img_feats["res_layer4_mean"]  # [B*N,512]
        if isinstance(img_feats, dict):
            img_feats = list([v for v in img_feats.values() if len(v.size()) == 4])

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))  # (B, N, C, H, W)

        return img_feats_reshaped, global_feat

    def _forward_impl(self, batch, **kwargs):
        img = batch["image"]  # (B, N, 3, H, W) 5 channels
        batch_size, num_cams = img.size(0), img.size(1)
        inp_img_shape = img.shape[-2:]
        img_feats, global_feat = self.extract_img_feat(img)  # [(B, N, C, H, W), ...]

        # prepare image_metas
        img_metas = {
            "inp_img_shape": inp_img_shape,  # h, w
            "cam_intr": batch["target_cam_intr"],  # tensor (B, N, 3, 3)
            "cam_extr": batch["target_cam_extr"],  # tensor  (B, N, 4, 4)
            # ...
        }

        template_pose = torch.zeros((1, 48)).to(img.device)
        template_betas = torch.zeros((1, 10)).to(img.device)
        mano_out = self.mano_layer(template_pose, template_betas)
        template_vertices = mano_out.verts
        template_3d_joints = mano_out.joints
        template_mesh = torch.cat([template_3d_joints, template_vertices], dim=1).squeeze(0)  # (799, 3)

        preds = self.head(
            mlvl_feats=img_feats,
            img_metas=img_metas,
            template_mesh=template_mesh,
            global_feat=global_feat,
        )

        # last decoder's output
        pred_joints_3d = preds["all_coords_preds"][-1, :, :self.num_joints, :]  # (B, 21, 3)
        pred_verts_3d = preds["all_coords_preds"][-1, :, self.num_joints:, :]  # (B, 778, 3)
        preds["pred_joints_3d"] = pred_joints_3d
        preds["pred_verts_3d"] = pred_verts_3d
        center_joint = pred_joints_3d[:, self.center_idx, :].unsqueeze(1)  # (B, 1, 3)
        preds["pred_joints_3d_rel"] = pred_joints_3d - center_joint
        preds["pred_verts_3d_rel"] = pred_verts_3d - center_joint
        return preds

    @staticmethod
    def loss_proj_to_multicam(pred_joints, T_c2m, K, gt_joints_2d, n_views, img_scale, tolerance_on_scale=1.0):
        pred_joints = pred_joints.unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 21, 3)
        pred_joints_in_cam = batch_cam_extr_transf(T_c2m, pred_joints)
        pred_joints_2d = batch_cam_intr_projection(K, pred_joints_in_cam)  # (B, N, 21, 2)
        multicam_proj_offset = torch.clamp(pred_joints_2d - gt_joints_2d,
                                           min=-tolerance_on_scale * img_scale,
                                           max=tolerance_on_scale * img_scale) / img_scale
        loss_2d_joints = torch.sum(torch.pow(multicam_proj_offset, 2), dim=3)  # (B, N, 21, 2)
        loss_2d_joints = torch.mean(loss_2d_joints)
        return loss_2d_joints

    def compute_loss(self, preds, gt):
        all_coords_preds = preds["all_coords_preds"]  # (N_Decoder, B, NUM_QUERY, 3)
        loss_dict = {}
        loss = 0
        batch_size = gt["image"].size(0)
        n_views = gt["image"].size(1)
        H = gt["image"].size(-2)
        W = gt["image"].size(-1)
        # use diagonal as scale
        img_scale = math.sqrt(float(W**2 + H**2))
        master_joints_gt = gt["master_joints_3d"]  # (B, 21, 3)
        master_verts_gt = gt["master_verts_3d"]  # (B, 778, 3)

        for i in range(all_coords_preds.shape[0]):
            # prepare
            gt_T_c2m = torch.linalg.inv(gt["target_cam_extr"])  # (B, N , 4, 4)
            pred_joints = all_coords_preds[i, :, :self.num_joints, :]  # (B, 21, 3)
            pred_verts = all_coords_preds[i, :, self.num_joints:, :]  # (B, 778, 3)
            pred_joints_from_mesh = mano_to_openpose(self.mano_layer.th_J_regressor, pred_verts)
            gt_joints_from_mesh = mano_to_openpose(self.mano_layer.th_J_regressor, master_verts_gt)

            gt_verts_ncams = master_verts_gt.unsqueeze(1).repeat(1, n_views, 1, 1)  # (B, N, 778, 3)
            gt_verts_ncams = batch_cam_extr_transf(gt_T_c2m, gt_verts_ncams)
            gt_verts_2d_ncams = batch_cam_intr_projection(gt["target_cam_intr"], gt_verts_ncams)  # (B, N, 778, 2)

            # 3D joints loss
            loss_3d_joints = self.criterion_joints(pred_joints, master_joints_gt)
            loss_3d_joints_from_mesh = self.criterion_joints(pred_joints_from_mesh, gt_joints_from_mesh)
            loss_recon = self.joints_weight * (loss_3d_joints + loss_3d_joints_from_mesh)

            # 3D verts loss
            loss_3d_verts = self.criterion_vertices(pred_verts, master_verts_gt)
            loss_recon += self.vertices_weight * loss_3d_verts

            # 2D joints loss
            if self.joints_2d_weight != 0:
                loss_2d_joints = self.loss_proj_to_multicam(pred_joints, gt_T_c2m, gt["target_cam_intr"],
                                                            gt["target_joints_2d"], n_views, img_scale)
            else:
                loss_2d_joints = torch.tensor(0.0).float().to(pred_verts.device)
            loss_recon += self.joints_2d_weight * loss_2d_joints

            # 2D verts loss
            if self.vertices_2d_weight != 0:
                loss_2d_verts = self.loss_proj_to_multicam(pred_verts, gt_T_c2m, gt["target_cam_intr"],
                                                           gt_verts_2d_ncams, n_views, img_scale)
            else:
                loss_2d_verts = torch.tensor(0.0).float().to(pred_verts.device)
            loss_recon += self.vertices_2d_weight * loss_2d_verts

            loss += loss_recon
            loss_dict[f'dec{i}_loss_recon'] = loss_recon

            if i == self.num_preds - 1:
                loss_dict[f'loss_3d_joints'] = loss_3d_joints
                loss_dict[f'loss_3d_joints_from_mesh'] = loss_3d_joints_from_mesh
                loss_dict[f'loss_3d_verts'] = loss_3d_verts
                loss_dict[f'loss_recon'] = loss_recon
                if self.joints_2d_weight != 0:
                    loss_dict[f'loss_2d_joints'] = loss_2d_joints
                if self.vertices_2d_weight != 0:
                    loss_dict[f"loss_2d_verts"] = loss_2d_verts

        loss_dict['loss'] = loss
        return loss, loss_dict

    def training_step(self, batch, step_idx, **kwargs):
        img = batch["image"]  # (B, N, 3, H, W) 5 channels
        batch_size = img.size(0)
        n_views = img.size(1)

        master_joints_3d = batch["master_joints_3d"]  # (B, 21, 3)
        master_verts_3d = batch["master_verts_3d"]  # (B, 778, 3)
        preds = self._forward_impl(batch, **kwargs)
        loss, loss_dict = self.compute_loss(preds, batch)

        ## last decoder's output
        pred_joints_3d = preds["pred_joints_3d"]  # (B, 21, 3)
        pred_verts_3d = preds["pred_verts_3d"]
        self.MPJPE_3D.feed(pred_joints_3d, gt_kp=master_joints_3d)
        self.MPVPE_3D.feed(pred_verts_3d, gt_kp=master_verts_3d)
        self.loss_metric.feed(loss_dict, batch_size)

        if step_idx % self.train_log_interval == 0:
            for k, v in loss_dict.items():
                self.summary.add_scalar(f"{k}", v.item(), step_idx)
            self.summary.add_scalar("MPJPE_3D", self.MPJPE_3D.get_result(), step_idx)
            self.summary.add_scalar("MPVPE_3D", self.MPVPE_3D.get_result(), step_idx)
            if step_idx % (self.train_log_interval * 5) == 0:  # viz every 10 * interval batches
                view_id = np.random.randint(n_views)
                img_toshow = img[:, view_id, ...]  # (B, 3, H, W)
                extr_toshow = torch.linalg.inv(batch["target_cam_extr"][:, view_id, ...])  # (B, 4, 4)s
                intr_toshow = batch["target_cam_intr"][:, view_id, ...]  # (B, 3, 3)

                pred_J3d_in_cam = (extr_toshow[:, :3, :3] @ pred_joints_3d.transpose(1, 2)).transpose(1, 2)
                pred_J3d_in_cam = pred_J3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)
                gt_J3d_in_cam = (extr_toshow[:, :3, :3] @ master_joints_3d.transpose(1, 2)).transpose(1, 2)
                gt_J3d_in_cam = gt_J3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)

                pred_J2d = batch_persp_project(pred_J3d_in_cam, intr_toshow)  # (B, 21, 2)
                gt_J2d = batch_persp_project(gt_J3d_in_cam, intr_toshow)  # (B, 21, 2)
                img_array = draw_batch_joint_images(pred_J2d, gt_J2d, img_toshow, step_idx, n_sample=4)
                self.summary.add_image(f"img/viz_joints_2d_train", img_array, step_idx, dataformats="NHWC")

                pred_V3d_in_cam = (extr_toshow[:, :3, :3] @ pred_verts_3d.transpose(1, 2)).transpose(1, 2)
                pred_V3d_in_cam = pred_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)
                gt_V3d_in_cam = (extr_toshow[:, :3, :3] @ master_verts_3d.transpose(1, 2)).transpose(1, 2)
                gt_V3d_in_cam = gt_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)

                pred_V2d = batch_persp_project(pred_V3d_in_cam, intr_toshow)  # (B, 21, 2)
                gt_V2d = batch_persp_project(gt_V3d_in_cam, intr_toshow)  # (B, 21, 2)
                img_array_verts = draw_batch_verts_images(pred_V2d, gt_V2d, img_toshow, step_idx, n_sample=4)
                self.summary.add_image(f"img/viz_verts_2d_train", img_array_verts, step_idx, dataformats="NHWC")

                face = self.mano_layer.th_faces.cpu().numpy()
                img_array_meshes = draw_batch_mesh_images(pred_V3d_in_cam,
                                                          gt_V3d_in_cam,
                                                          face=face,
                                                          intr=intr_toshow,
                                                          tensor_image=img_toshow,
                                                          step_idx=step_idx,
                                                          n_sample=4)
                self.summary.add_image(f"img/viz_mesh_train", img_array_meshes, step_idx, dataformats="NHWC")

        return preds, loss_dict

    def on_train_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-train"
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([self.MPJPE_3D, self.MPVPE_3D], epoch_idx, comment=comment)
        self.loss_metric.reset()
        self.MPJPE_3D.reset()
        self.MPVPE_3D.reset()

    def validation_step(self, batch, step_idx, **kwargs):
        preds = self.testing_step(batch, step_idx, **kwargs)
        img = batch["image"]  # (B, N, 3, H, W) 5 channels
        n_views = img.size(1)
        pred_joints_3d = preds["pred_joints_3d"]  # (B, 21, 3)
        master_joints_gt = batch["master_joints_3d"]  # (B, 21, 3)

        pred_verts_3d = preds["pred_verts_3d"]
        master_verts_3d = batch["master_verts_3d"]  # (B, 778, 3)

        self.summary.add_scalar("MPJPE_3D_val", self.MPJPE_3D.get_result(), step_idx)
        self.summary.add_scalar("MPVPE_3D_val", self.MPVPE_3D.get_result(), step_idx)
        if step_idx % (self.train_log_interval * 10) == 0:
            view_id = np.random.randint(n_views)
            img_toshow = img[:, view_id, ...]  # (B, 3, H, W)
            extr_toshow = torch.linalg.inv(batch["target_cam_extr"][:, view_id, ...])  # (B, 4, 4)
            intr_toshow = batch["target_cam_intr"][:, view_id, ...]  # (B, 3, 3)

            pred_J3d_in_cam = (extr_toshow[:, :3, :3] @ pred_joints_3d.transpose(1, 2)).transpose(1, 2)
            pred_J3d_in_cam = pred_J3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)
            gt_J3d_in_cam = (extr_toshow[:, :3, :3] @ master_joints_gt.transpose(1, 2)).transpose(1, 2)
            gt_J3d_in_cam = gt_J3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)

            pred_J2d = batch_persp_project(pred_J3d_in_cam, intr_toshow)  # (B, 21, 2)
            gt_J2d = batch_persp_project(gt_J3d_in_cam, intr_toshow)  # (B, 21, 2)
            img_array = draw_batch_joint_images(pred_J2d, gt_J2d, img_toshow, step_idx, n_sample=4)
            self.summary.add_image(f"img/viz_joints_2d_val", img_array, step_idx, dataformats="NHWC")

            pred_V3d_in_cam = (extr_toshow[:, :3, :3] @ pred_verts_3d.transpose(1, 2)).transpose(1, 2)
            pred_V3d_in_cam = pred_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)
            gt_V3d_in_cam = (extr_toshow[:, :3, :3] @ master_verts_3d.transpose(1, 2)).transpose(1, 2)
            gt_V3d_in_cam = gt_V3d_in_cam + extr_toshow[:, :3, 3].unsqueeze(1)

            pred_V2d = batch_persp_project(pred_V3d_in_cam, intr_toshow)  # (B, 21, 2)
            gt_V2d = batch_persp_project(gt_V3d_in_cam, intr_toshow)  # (B, 21, 2)
            img_array_verts = draw_batch_verts_images(pred_V2d, gt_V2d, img_toshow, step_idx, n_sample=4)
            self.summary.add_image(f"img/viz_verts_2d_val", img_array_verts, step_idx, dataformats="NHWC")

            face = self.mano_layer.th_faces.cpu().numpy()
            img_array_meshes = draw_batch_mesh_images(pred_V3d_in_cam,
                                                      gt_V3d_in_cam,
                                                      face=face,
                                                      intr=intr_toshow,
                                                      tensor_image=img_toshow,
                                                      step_idx=step_idx,
                                                      n_sample=4)
            self.summary.add_image(f"img/viz_mesh_val", img_array_meshes, step_idx, dataformats="NHWC")

        return preds

    def on_val_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-val"
        recorder.record_metric([
            self.MPJPE_3D,
            self.MPVPE_3D,
            self.MPJPE_3D_REL,
            self.MPVPE_3D_REL,
            self.PA,
        ],
                               epoch_idx,
                               comment=comment)
        self.MPJPE_3D.reset()
        self.MPVPE_3D.reset()
        self.MPJPE_3D_REL.reset()
        self.MPVPE_3D_REL.reset()
        self.PA.reset()

    def testing_step(self, batch, step_idx, **kwargs):

        img = batch["image"]  # (B, N, 3, H, W) 5 channels
        batch_size = img.size(0)
        n_views = img.size(1)

        preds = self._forward_impl(batch, **kwargs)
        pred_verts_3d = preds["pred_verts_3d"]  # (B, 21, 3)
        gt_verts_3d = batch["master_verts_3d"]  # (B, 778, 3)

        if self.pred_joints_from_mesh is True:
            gt_joints_3d = mano_to_openpose(self.mano_layer.th_J_regressor, gt_verts_3d)
            pred_joints_3d = mano_to_openpose(self.mano_layer.th_J_regressor, pred_verts_3d)
        else:
            gt_joints_3d = batch["master_joints_3d"]  # (B, 21, 3)
            pred_joints_3d = preds["pred_joints_3d"]  # (B, 21, 3)

        self.MPJPE_3D.feed(pred_joints_3d, gt_kp=gt_joints_3d)
        self.MPVPE_3D.feed(pred_verts_3d, gt_kp=gt_verts_3d)

        pred_joints_3d_rel = pred_joints_3d - pred_joints_3d[:, self.center_idx, :].unsqueeze(1)
        gt_joints_3d_rel = gt_joints_3d - gt_joints_3d[:, self.center_idx, :].unsqueeze(1)
        self.MPJPE_3D_REL.feed(pred_joints_3d_rel, gt_kp=gt_joints_3d_rel)

        pred_verts_3d_rel = pred_verts_3d - pred_joints_3d[:, self.center_idx, :].unsqueeze(1)
        gt_verts_3d_rel = gt_verts_3d - gt_joints_3d[:, self.center_idx, :].unsqueeze(1)
        self.MPVPE_3D_REL.feed(pred_verts_3d_rel, gt_kp=gt_verts_3d_rel)

        self.PA.feed(pred_joints_3d, gt_joints_3d, pred_verts_3d, gt_verts_3d)

        if "callback" in kwargs:
            callback = kwargs.pop("callback")
            if callable(callback):
                callback(preds, batch, step_idx, **kwargs)

        return preds

    def on_test_finished(self, recorder: Recorder, epoch_idx, **kwargs):
        comment = f"{self.name}-test"
        recorder.record_metric([
            self.MPJPE_3D,
            self.MPVPE_3D,
            self.MPJPE_3D_REL,
            self.MPVPE_3D_REL,
            self.PA,
        ],
                               epoch_idx,
                               comment=comment)
        self.MPJPE_3D.reset()
        self.MPVPE_3D.reset()
        self.MPJPE_3D_REL.reset()
        self.MPVPE_3D_REL.reset()
        self.PA.reset()

    def format_metric(self, mode="train"):
        if mode == "train":
            metric_str = (f"3DV {self.loss_metric.get_loss('loss_3d_verts'):.5f} | "
                          f"3DJ {self.loss_metric.get_loss('loss_3d_joints'):.5f} | ")
            return metric_str

        elif mode == "test":
            metric_toshow = [self.PA]
        else:
            metric_toshow = [self.MPJPE_3D, self.MPVPE_3D]

        return " | ".join([str(me) for me in metric_toshow])

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
