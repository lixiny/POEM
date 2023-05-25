import copy
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.metrics import LossMetric, MeanEPE, PAEval
from lib.models.layers.mano_wrapper import MANO, kpId2vertices
from lib.models.model_abstraction import ModuleAbstract
from lib.utils.builder import MODEL
from lib.utils.logger import logger
from lib.utils.misc import param_size
from lib.utils.net_utils import init_weights
from lib.utils.transform import batch_persp_project, mano_to_openpose
from lib.viztools.draw import draw_batch_joint_images

from .loss import bce_loss, edge_length_loss, l1_loss, normal_loss
from .net import ConvBlock, ParallelDeblock, Pool, SelfAttention, SpiralConv
from .data_adaptor import map2uv, uv2map
from .regitstration import registration_one, cnt_area


class EncodeUV(nn.Module):

    def __init__(self, backbone):
        super(EncodeUV, self).__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x0, x4, x3, x2, x1


class EncodeMesh(nn.Module):

    def __init__(self, backbone, in_channel):
        super(EncodeMesh, self).__init__()
        self.reduce = nn.Sequential(ConvBlock(in_channel, in_channel, relu=True, norm='bn'),
                                    ConvBlock(in_channel, 128, relu=True, norm='bn'),
                                    ConvBlock(128, 64, kernel_size=1, padding=0, relu=False, norm='bn'))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x = self.reduce(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, x4, x3, x2, x1


@MODEL.register_module()
class CMR_G(nn.Module, ModuleAbstract):
    """
    Implementation of CMR_SG.
    :param spiral_indices: pre-defined spiral sample
    :param up_transform: pre-defined upsample matrix
    :param relation: This implementation only adopts tip-based aggregation.
                     You can employ more sub-poses by enlarge relation list.
    """

    def __init__(self, cfg):
        super(CMR_G, self).__init__()

        self.name = 'CMR_G'
        self.cfg = cfg
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.has_spiral_transform = False
        self.V_STD = 0.2
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.img_size = cfg.DATA_PRESET.IMAGE_SIZE

        self.J_regressor = MANO(
            mano_assets_root="assets/mano_v1_2",
            center_idx=None,
        ).mano_layer.th_J_regressor

        self.PA = PAEval(cfg.METRIC)
        self.MPJPE_CS = MeanEPE(cfg, "J_cs")
        self.MPVPE_CS = MeanEPE(cfg, "V_cs")

        args = cfg
        self.in_channels = args.IN_CHANNELS
        self.out_channels = args.OUT_CHANNELS

        self.loss_metric = LossMetric(cfg)

        ## currnently, we cannnot store a ref (self.) to the spiral_indices and up_transform
        # self.spiral_indices = spiral_indices
        # self.up_transform = up_transform
        work_dir = os.path.dirname(os.path.realpath(__file__))
        self.transf_pth = os.path.join(work_dir, 'template', 'transform.pkl')
        self.template_pth = os.path.join(work_dir, 'template', 'template.ply')

        from .utils import spiral_tramsform
        spiral_indices, _, up_transform, tmp = spiral_tramsform(self.transf_pth, self.template_pth)
        self.has_spiral_transform = False

        self.num_vert = [u.size(0) for u in up_transform] + [up_transform[-1].size(1)]
        self.uv_channel = 21
        self.relation = [
            [4, 8],
            [4, 12],
            [4, 16],
            [4, 20],
            [8, 12],
            [8, 16],
            [8, 20],
            [12, 16],
            [12, 20],
            [16, 20],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
        ]
        self.att: bool = args.ATT

        backbone, self.latent_size = self.get_backbone(args.BACKBONE)
        self.backbone = EncodeUV(backbone)

        backbone2, _ = self.get_backbone(args.BACKBONE)
        self.backbone_mesh = EncodeMesh(backbone2, 64 + self.uv_channel + len(self.relation))

        self.uv_delayer = nn.ModuleList([
            ConvBlock(self.latent_size[2] + self.latent_size[1],
                      self.latent_size[2],
                      kernel_size=3,
                      relu=True,
                      norm='bn'),
            ConvBlock(self.latent_size[3] + self.latent_size[2],
                      self.latent_size[3],
                      kernel_size=3,
                      relu=True,
                      norm='bn'),
            ConvBlock(self.latent_size[4] + self.latent_size[3],
                      self.latent_size[4],
                      kernel_size=3,
                      relu=True,
                      norm='bn'),
            ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
        ])
        self.uv_head = ConvBlock(self.latent_size[4], self.uv_channel, kernel_size=3, padding=1, relu=False, norm=None)

        self.uv_delayer2 = nn.ModuleList([
            ConvBlock(self.latent_size[2] + self.latent_size[1],
                      self.latent_size[2],
                      kernel_size=3,
                      relu=True,
                      norm='bn'),
            ConvBlock(self.latent_size[3] + self.latent_size[2],
                      self.latent_size[3],
                      kernel_size=3,
                      relu=True,
                      norm='bn'),
            ConvBlock(self.latent_size[4] + self.latent_size[3],
                      self.latent_size[4],
                      kernel_size=3,
                      relu=True,
                      norm='bn'),
            ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
        ])
        self.uv_head2 = ConvBlock(self.latent_size[4],
                                  self.uv_channel + 1,
                                  kernel_size=3,
                                  padding=1,
                                  relu=False,
                                  norm=None)

        # 3D decoding
        if self.att:
            self.attention = SelfAttention(self.latent_size[0])
        self.de_layers = nn.ModuleList()
        self.de_layers.append(nn.Linear(self.latent_size[0], self.num_vert[-1] * self.out_channels[-1]))
        for idx in range(len(self.out_channels)):
            if idx == 0:
                self.de_layers.append(
                    ParallelDeblock(self.out_channels[-idx - 1], self.out_channels[-idx - 1], spiral_indices[-idx - 1]))
            else:
                self.de_layers.append(
                    ParallelDeblock(self.out_channels[-idx] + 3, self.out_channels[-idx - 1], spiral_indices[-idx - 1]))
        self.heads = nn.ModuleList()
        for oc, sp_idx in zip(self.out_channels[::-1], spiral_indices[::-1]):
            self.heads.append(SpiralConv(oc, self.in_channels, sp_idx))

        if cfg.PRETRAINED:
            init_weights(self, pretrained=cfg.PRETRAINED)
            # self.init_weights(pretrained=cfg.PRETRAINED)

        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def get_backbone(self, backbone, pretrained=True):
        from lib.models.backbones.resnet import resnet18, resnet34, resnet50
        if '50' in backbone:
            basenet = resnet50(pretrained=pretrained)
            latent_channel = (1000, 2048, 1024, 512, 256)
        elif '34' in backbone:
            basenet = resnet34(pretrained=pretrained)
            latent_channel = (1000, 512, 256, 128, 64)
        elif '18' in backbone:
            basenet = resnet18(pretrained=pretrained)
            latent_channel = (1000, 512, 256, 128, 64)
        else:
            raise Exception("Not supported", backbone)

        return basenet, latent_channel

    def decoder(self, x):
        if self.att:
            x = self.attention(x)
        num_layers = len(self.de_layers)
        num_features = num_layers - 1
        hierachy_pred = []
        for i, layer in enumerate(self.de_layers):
            if i == 0:
                x = layer(x)
                x = x.view(-1, self.num_vert[-1], self.out_channels[-1])
            else:
                x = layer(x, self.up_transform[num_features - i])
                pred = self.heads[i - 1](x)
                if i > 1:
                    pred = (pred + Pool(hierachy_pred[-1], self.up_transform[num_features - i])) / 2
                hierachy_pred.append(pred)
                x = torch.cat((x, pred), 2)

        return hierachy_pred[::-1]

    def uv_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.uv_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i + 1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uv_head(x))

        return pred

    def uv_decoder2(self, z):
        x = z[0]
        for i, de in enumerate(self.uv_delayer2):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i + 1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uv_head2(x))

        return pred

    def setup(self, summary_writer, **kwargs):
        self.summary = summary_writer

    def _forward_impl(self, data):
        if not self.has_spiral_transform:
            from .utils import spiral_tramsform
            spiral_indices, _, up_transform, tmp = spiral_tramsform(self.transf_pth, self.template_pth)

            self.spiral_indices = spiral_indices
            self.up_transform = up_transform
            self.face = torch.from_numpy(tmp['face'][0].astype(np.int64))
            self.has_spiral_transform = True

        x = data['img']
        z_uv = self.backbone(x)

        uv_prior = self.uv_decoder(z_uv[1:])
        z_mesh = self.backbone_mesh(
            torch.cat([z_uv[0], uv_prior] + [uv_prior[:, i].sum(dim=1, keepdim=True) for i in self.relation], 1))
        pred = self.decoder(z_mesh[0])
        uv = self.uv_decoder2(z_mesh[1:])

        return {
            'mesh_pred': pred,
            'uv_pred': uv[:, :self.uv_channel],
            'mask_pred': uv[:, self.uv_channel],
            'uv_prior': uv_prior,
        }

    def training_step(self, data, step_idx, **kwargs):
        batch_size = data["img"].shape[0]
        out = self._forward_impl(data)

        loss, loss_dict = self.compute_loss(pred=out['mesh_pred'],
                                            gt=data.get('mesh_gt'),
                                            uv_pred=out.get('uv_pred'),
                                            uv_gt=data.get('uv_gt'),
                                            mask_pred=out.get('mask_pred'),
                                            mask_gt=data.get('mask_gt'),
                                            face=self.face,
                                            uv_prior=out.get('uv_prior'),
                                            uv_prior2=out.get('uv_prior2'),
                                            mask_prior=out.get('mask_prior'))

        self.loss_metric.feed(loss_dict, batch_size)
        if step_idx % self.train_log_interval == 0:
            self.summary.add_scalar("loss", loss.item(), step_idx)
            self.summary.add_scalar("mesh_l1_loss", loss_dict["l1_loss"].item(), step_idx)
            self.summary.add_scalar("uv_loss", loss_dict["uv_loss"].item(), step_idx)
            self.summary.add_scalar("uv_prior_loss", loss_dict["uv_prior_loss"].item(), step_idx)

            viz_interval = self.train_log_interval * 20
            if step_idx % viz_interval == 0:
                self.board_img('train',
                               step_idx,
                               data['img'][0],
                               mask_gt=data.get('mask_gt'),
                               mask_pred=out.get('mask_pred'),
                               uv_gt=data.get('uv_gt'),
                               uv_pred=out.get('uv_pred'),
                               uv_prior=out.get('uv_prior'))

        return out, loss_dict

    def board_img(self, phase, n_iter, img, **kwargs):
        from .utils import tensor2array

        # print(rendered_mask.shape, rendered_mask.max(), rendered_mask.min())
        self.summary.add_image(phase + '/img', tensor2array(img), n_iter)
        if kwargs.get('mask_pred') is not None:
            self.summary.add_image(phase + '/mask_gt', tensor2array(kwargs['mask_gt'][0]), n_iter)
            self.summary.add_image(phase + '/mask_pred', tensor2array(kwargs['mask_pred'][0]), n_iter)
        if kwargs.get('uv_pred') is not None:
            self.summary.add_image(phase + '/uv_gt', tensor2array(kwargs['uv_gt'][0].sum(dim=0).clamp(max=1)), n_iter)
            self.summary.add_image(phase + '/uv_pred', tensor2array(kwargs['uv_pred'][0].sum(dim=0).clamp(max=1)),
                                   n_iter)
        if kwargs.get('uv_prior') is not None:
            self.summary.add_image(phase + '/uv_prior', tensor2array(kwargs['uv_prior'][0].sum(dim=0).clamp(max=1)),
                                   n_iter)

    def compute_loss(self, **kwargs):
        loss_dict = dict()
        loss = 0.
        for i in range(len(kwargs['gt'])):
            loss += l1_loss(kwargs['pred'][i], kwargs['gt'][i])
            if i == 0:
                loss_dict['l1_loss'] = loss.clone()
        loss_dict['uv_loss'] = 10 * bce_loss(kwargs['uv_pred'], kwargs['uv_gt'])
        loss_dict['uv_prior_loss'] = 10 * bce_loss(kwargs['uv_prior'], kwargs['uv_gt'])
        loss_dict['mask_loss'] = 0.5 * bce_loss(kwargs['mask_pred'], kwargs['mask_gt'])
        loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])
        loss_dict['edge_loss'] = edge_length_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])

        loss += loss_dict['uv_loss'] + loss_dict['normal_loss'] + loss_dict['edge_loss'] + \
                loss_dict['uv_prior_loss'] + loss_dict['mask_loss']
        loss_dict['loss'] = loss

        return loss, loss_dict

    def on_train_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-train-"  # CMR_G-train
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        self.loss_metric.reset()

    def validation_step(self, data, step_idx, **kwargs):
        BATCH_SIZE = data["img"].shape[0]
        out = self._forward_impl(data)

        mesh_gt = data['mesh_gt'][0] if isinstance(data['mesh_gt'], list) else data['mesh_gt']
        mesh_gt = mesh_gt * self.V_STD
        xyz_gt = data['xyz_gt'] * self.V_STD

        mesh_pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
        mesh_pred = mesh_pred * self.V_STD
        mesh_pred = mesh_pred.detach().cpu()

        joint_pred = mano_to_openpose(self.J_regressor, mesh_pred)

        self.PA.feed(joint_pred, xyz_gt, mesh_pred, mesh_gt)

        res = {"final_mesh_pred": mesh_pred, "final_joint_pred": joint_pred, **out}
        return res

    def on_val_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-val-"
        recorder.record_metric([self.PA], epoch_idx, comment=comment)
        self.PA.reset()
        return

    def testing_step(self, data, step_idx, **kwargs):
        img = data['img']  # (B, 3, H, W)
        BATCH_SIZE = img.shape[0]
        out = self.validation_step(data, step_idx)

        with_registration = kwargs.get("with_registration", True)
        if with_registration:
            for i in range(BATCH_SIZE):
                mesh_pred = out['final_mesh_pred'][i].numpy()
                joint_pred = out['final_joint_pred'][i].numpy()
                uv_pred = out['uv_pred'][i].unsqueeze(0)  # (1, 21, 128, 128)
                assert uv_pred.ndim == 4, f"wrong uv_pred shape: {uv_pred.shape}"
                uv_point_pred, uv_pred_conf = map2uv(uv_pred.cpu().numpy(), (img.size(2), img.size(3)))

                K = data['K'][i].cpu().numpy()
                mask_pred = out['mask_pred'][i]  # (128, 128)
                mask_pred = (mask_pred > 0.3).cpu().numpy().astype(np.uint8)
                mask_pred = cv2.resize(mask_pred, (img.size(3), img.size(2)))  # (256, 256)

                try:
                    contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours.sort(key=cnt_area, reverse=True)
                    poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                except:
                    poly = None

                xyz_root = data['xyz_root'][i].detach().cpu().unsqueeze(0)  # (1, 3)
                mesh_gt = data['mesh_gt'][0] if isinstance(data['mesh_gt'], list) else data['mesh_gt']
                mesh_gt = mesh_gt[i].detach().cpu() * self.V_STD + xyz_root
                xyz_gt = data['xyz_gt'][i].detach().cpu() * self.V_STD + xyz_root

                vertex, t, success = registration_one(vertex=mesh_pred,
                                                      vertex2xyz=joint_pred,
                                                      uv=uv_point_pred[0],
                                                      uv_conf=uv_pred_conf[0],
                                                      K=K,
                                                      size=self.img_size[1],
                                                      poly=poly)
                if success is False:
                    # logger.error(f"registration failed: at step: {step_idx} sample {i}")
                    continue

                vertex = torch.from_numpy(vertex[None, ...]).float()
                joints = mano_to_openpose(self.J_regressor, vertex)

                self.MPJPE_CS.feed(pred_kp=joints, gt_kp=xyz_gt.unsqueeze(0))
                self.MPVPE_CS.feed(pred_kp=vertex, gt_kp=mesh_gt.unsqueeze(0))

        if "callback" in kwargs:
            kwargs["callback"](preds=out, inputs=data, step_idx=step_idx)

        return out

    def _testing_step(self, data, step_idx, **kwargs):
        BATCH_SIZE = data["img"].shape[0]
        out = self._forward_impl(data)

        mesh_gt = data['mesh_gt'][0] if isinstance(data['mesh_gt'], list) else data['mesh_gt']
        mesh_gt = mesh_gt * self.V_STD
        xyz_gt = data['xyz_gt'] * self.V_STD

        mesh_pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
        mesh_pred = mesh_pred * self.V_STD
        mesh_pred = mesh_pred.detach().cpu()
        joint_pred = torch.matmul(self.J_regressor, mesh_pred)

        tipsId = [v[0] for k, v in kpId2vertices.items()]
        tips = mesh_pred[:, tipsId]
        joint_pred = torch.cat([joint_pred, tips], dim=1)
        # Reorder joints to match OpenPose definition
        joint_pred = joint_pred[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
        # root_j = joint_pred[:, self.center_idx].unsqueeze(1)
        self.PA.feed(joint_pred, xyz_gt, mesh_pred, mesh_gt)

        # xyz_root = data["xyz_root"].cpu()  # (B, 3)
        # joint_pred_CS = joint_pred + xyz_root.unsqueeze(1)  # (B, 21, 3)
        # K = data["K"].cpu()  # (3, 3)
        # joints_proj = batch_persp_project(joint_pred_CS, K)  # (B, 21, 2)
        # img_array = draw_batch_joint_images(joints_proj, data["uv_point"], data["img"], step_idx)
        # self.summary.add_images("img/test_joints2d", img_array, step_idx, dataformats="NHWC")

    def on_test_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-test-"
        recorder.record_metric([self.PA, self.MPJPE_CS, self.MPVPE_CS], epoch_idx, comment=comment)
        self.PA.reset()
        self.MPVPE_CS.reset()
        self.MPJPE_CS.reset()
        return

    def format_metric(self, mode="train"):
        if mode == "val":
            metric_toshow = [self.PA]
            return " | ".join([str(me) for me in metric_toshow])
        elif mode == "test":
            metric_toshow = [self.PA, self.MPJPE_CS, self.MPVPE_CS]
            return " | ".join([str(me) for me in metric_toshow])
        elif mode == "train":
            return f"{self.loss_metric.get_loss('loss'):.4f}"
        else:
            return ""

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
