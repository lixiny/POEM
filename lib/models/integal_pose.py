import os
import warnings
from collections import OrderedDict
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics.basic_metric import LossMetric
from ..metrics.pa_eval import PAEval
from ..metrics.mean_epe import MeanEPE
from ..utils.builder import MODEL
from ..utils.logger import logger
from ..utils.misc import CONST, enable_lower_param, param_size
from ..utils.net_utils import init_weights
from ..utils.transform import batch_uvd2xyz
from ..viztools.draw import draw_batch_joint_images
from .backbones import create_backbone
from .model_abstraction import ModuleAbstract


@MODEL.register_module()
class IntegralPose(nn.Module, ModuleAbstract):

    @enable_lower_param
    def __init__(self, cfg):
        super(IntegralPose, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg
        self.train_log_interval = cfg.TRAIN.LOG_INTERVAL
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        self.inp_res = cfg.DATA_PRESET.IMAGE_SIZE
        self.train_mode = cfg.MODE
        assert self.train_mode in ["3D", "UVD_ortho", "UVD"], f"Model's mode mismatch, got {self.train_mode}"
        self.loss_metric = LossMetric(cfg)
        self.mepe3d_metric = MeanEPE(cfg, "joints_3d")
        self.mepe2d_metric = MeanEPE(cfg, "joints_2d")
        self.PA_metric = PAEval(cfg.METRIC)

        # ***** build network *****
        self.backbone = create_backbone(cfg.BACKBONE)
        self.pose_head = IntegralDeconvHead(cfg.HEAD)

        if cfg.BACKBONE.PRETRAINED and cfg.PRETRAINED:
            logger.warning(f"{self.name}'s backbone {cfg.BACKBONE.TYPE} re-initalized by {cfg.PRETRAINED}")
        init_weights(self, pretrained=cfg.PRETRAINED)
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

    def compute_loss(self, preds, gt):
        gt_joints_vis = gt["target_joints_vis"]  # (B, NJ)
        gt_uvd = gt["target_joints_uvd"]  # (B, NJ, 3)
        pred_uvd = preds["uvd"]  # (B, NJ, 3)
        pred_uvd = torch.einsum("bij,bi->bij", pred_uvd, gt_joints_vis)  # (B, NJ, 3)
        gt_uvd = torch.einsum("bij,bi->bij", gt_uvd, gt_joints_vis)  # (B, NJ, 3)

        if self.cfg.LOSS.TYPE == "mse":
            uvd_loss = F.mse_loss(pred_uvd, gt_uvd)
        elif self.cfg.LOSS.TYPE == "smooth_l1":
            uvd_loss = F.smooth_l1_loss(pred_uvd, gt_uvd)

        total_loss = self.cfg.LOSS.LAMBDA_UVD * uvd_loss
        return {"loss": total_loss, "uvd_loss": uvd_loss}

    def validation_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch)
        return preds, {}

    def training_step(self, batch, step_idx, **kwargs):
        # forward your network
        preds = self._forward_impl(batch)

        # compute loss
        loss_dict = self.compute_loss(preds, batch)

        batch_size = batch["image"].shape[0]
        self.loss_metric.feed(loss_dict, batch_size)

        joints_uvd = preds["uvd"]
        inp_res = torch.Tensor(self.inp_res).to(joints_uvd.device)
        joints_2d = torch.einsum("bij,j->bij", joints_uvd[:, :, :2], inp_res)
        preds["joints_2d"] = joints_2d

        if self.cfg.MODE in ["3D", "UVD_ortho"]:
            cam_mode = "ortho" if self.cfg.MODE == "UVD_ortho" else "persp"
            intr = batch["target_ortho_intr"] if cam_mode == "ortho" else batch["target_cam_intr"]
            joints_3d = batch_uvd2xyz(uvd=joints_uvd,
                                      root_joint=batch["target_root_joint"],
                                      intr=intr,
                                      inp_res=self.inp_res,
                                      depth_range=CONST.UVD_DEPTH_RANGE,
                                      camera_mode=cam_mode)
            preds["joints_3d"] = joints_3d
            self.mepe3d_metric.feed(joints_3d, batch["target_joints_3d"])
        elif self.cfg.MODE == "UVD":
            # YT3D only has gt annotaiton of UV & D, no xyz.
            self.mepe3d_metric.feed(joints_uvd, batch["target_joints_uvd"])

        if step_idx % self.train_log_interval == 0:
            self.summary.add_scalar("uvd_loss", loss_dict['uvd_loss'].item(), step_idx)
            self.summary.add_scalar("j3d_mepe", self.mepe3d_metric.get_result(), step_idx)

            img_array = draw_batch_joint_images(joints_2d, batch["target_joints_2d"], batch["image"], step_idx)
            self.summary.add_images("joints2d", img_array, step_idx, dataformats="NHWC")

        return preds, loss_dict

    def on_train_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-train-{self.train_mode}"  # SimpleBaseline3D-train-3D
        recorder.record_loss(self.loss_metric, epoch_idx, comment=comment)
        recorder.record_metric([self.mepe3d_metric], epoch_idx, comment=comment)

        self.loss_metric.reset()
        self.mepe3d_metric.reset()

    def on_val_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-val-{self.train_mode}"  # SimpleBaseline3D-train-3D
        pass

    def testing_step(self, batch, step_idx, **kwargs):
        preds = self._forward_impl(batch)
        batch_size = batch["image"].shape[0]
        joints_uvd = preds["uvd"]
        inp_res = torch.Tensor(self.inp_res).to(joints_uvd.device)
        joints_2d = torch.einsum("bij,j->bij", joints_uvd[:, :, :2], inp_res)
        preds["joints_2d"] = joints_2d

        if self.cfg.MODE in ["3D", "UVD_ortho"]:
            camera_mode = "ortho" if self.cfg.MODE == "UVD_ortho" else "persp"
            intr = batch["target_ortho_intr"] if camera_mode == "ortho" else batch["target_cam_intr"]
            joints_3d = batch_uvd2xyz(uvd=joints_uvd,
                                      root_joint=batch["target_root_joint"],
                                      intr=intr,
                                      inp_res=self.inp_res,
                                      depth_range=CONST.UVD_DEPTH_RANGE,
                                      camera_mode=camera_mode)
            preds["joints_3d"] = joints_3d
            self.mepe3d_metric.feed(joints_3d, batch["target_joints_3d"])
            self.PA_metric.feed(joints_3d, batch["target_joints_3d"])
        elif self.cfg.MODE == "UVD":
            # some dataset (eg. YT3D) only has gt annotaiton of UV&D, no xyz.
            self.mepe3d_metric.feed(joints_uvd, batch["target_joints_uvd"])

    def on_test_finished(self, recorder, epoch_idx):
        comment = f"{self.name}-test-{self.train_mode}"
        recorder.record_metric([self.PA_metric], epoch_idx, comment=comment)
        self.mepe3d_metric.reset()
        self.PA_metric.reset()
        return

    def _forward_impl(self, inputs):
        x = inputs["image"]
        feat = self.backbone(image=x)
        res = self.pose_head(feature=feat["res_layer4"])
        return res

    def init_weights(self, pretrained=""):
        warnings.warn(f"`module.init_weights` is deprecated and will be removed, "
                      f"use `init_weights(self, pretrianed)` from net_uilts instead.")

        if pretrained == "":
            logger.warning(f"=> Init {self.name} weights in backbone and head")
            """
            Add init for other modules
            ...
            """
        elif os.path.isfile(pretrained):
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
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"=> Loading SUCCEEDED")
        else:
            logger.error(f"=> No {self.name} checkpoints file found in {pretrained}")
            raise FileNotFoundError()



def norm_heatmap(norm_type: str, heatmap: torch.Tensor) -> torch.Tensor:
    """
    Args:
        norm_type: str: either in [softmax, sigmoid, divide_sum],
        heatmap: TENSOR (BATCH, C, ...)

    Returns:
        TENSOR (BATCH, C, ...)
    """
    shape = heatmap.shape
    if norm_type == "softmax":
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_type == "sigmoid":
        return heatmap.sigmoid()
    else:
        raise NotImplementedError


def integral_heatmap2d(heatmap2d: torch.Tensor) -> torch.Tensor:
    """
    Integral 2D heatmap into wh corrdinates. u stand for the prediction in WIDTH dimension
    ref: https://arxiv.org/abs/1711.08229

    Args:
        heatmap2d: TENSOR (BATCH, NCLASSES, HEIGHT, WIDTH) v,u

    Returns:
        uvd: TENSOR (BATCH, NCLASSES, 2) RANGE:0~1
    """
    # d_accu = torch.sum(heatmap3d, dim=[3, 4])
    v_accu = torch.sum(heatmap2d, dim=3)  # (B, C, H)
    u_accu = torch.sum(heatmap2d, dim=2)  # (B, C, W)

    weightv = torch.arange(v_accu.shape[-1], dtype=v_accu.dtype, device=v_accu.device) / v_accu.shape[-1]
    weightu = torch.arange(u_accu.shape[-1], dtype=u_accu.dtype, device=u_accu.device) / u_accu.shape[-1]

    v_ = v_accu.mul(weightv)
    u_ = u_accu.mul(weightu)

    v_ = torch.sum(v_, dim=-1, keepdim=True)
    u_ = torch.sum(u_, dim=-1, keepdim=True)

    uv = torch.cat([u_, v_], dim=-1)
    return uv  # TENSOR (BATCH, NCLASSES, 2)


def integral_heatmap3d(heatmap3d: torch.Tensor) -> torch.Tensor:
    """
    Integral 3D heatmap into whd corrdinates. u stand for the prediction in WIDTH dimension
    ref: https://arxiv.org/abs/1711.08229

    Args:
        heatmap3d: TENSOR (BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH) d,v,u

    Returns:
        uvd: TENSOR (BATCH, NCLASSES, 3) RANGE:0~1
    """
    d_accu = torch.sum(heatmap3d, dim=[3, 4])
    v_accu = torch.sum(heatmap3d, dim=[2, 4])
    u_accu = torch.sum(heatmap3d, dim=[2, 3])

    weightd = torch.arange(d_accu.shape[-1], dtype=d_accu.dtype, device=d_accu.device) / d_accu.shape[-1]
    weightv = torch.arange(v_accu.shape[-1], dtype=v_accu.dtype, device=v_accu.device) / v_accu.shape[-1]
    weightu = torch.arange(u_accu.shape[-1], dtype=u_accu.dtype, device=u_accu.device) / u_accu.shape[-1]

    d_ = d_accu.mul(weightd)
    v_ = v_accu.mul(weightv)
    u_ = u_accu.mul(weightu)

    d_ = torch.sum(d_, dim=-1, keepdim=True)
    v_ = torch.sum(v_, dim=-1, keepdim=True)
    u_ = torch.sum(u_, dim=-1, keepdim=True)

    uvd = torch.cat([u_, v_, d_], dim=-1)
    return uvd  # TENSOR (BATCH, NCLASSES, 3)


class IntegralDeconvHead(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.inplanes = cfg.INPUT_CHANNEL
        self.depth_res = cfg.HEATMAP_3D_SIZE[2]
        self.height_res = cfg.HEATMAP_3D_SIZE[1]
        self.width_res = cfg.HEATMAP_3D_SIZE[0]
        self.deconv_with_bias = cfg.DECONV_WITH_BIAS
        self.nclasses = cfg.N_CLASSES
        self.norm_type = cfg.NORM_TYPE

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            cfg.NUM_DECONV_LAYERS,
            cfg.NUM_DECONV_FILTERS,
            cfg.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.N_CLASSES * self.depth_res,
            kernel_size=cfg.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.FINAL_CONV_KERNEL == 3 else 0,
        )
        self.init_weights()

    def init_weights(self):
        logger.info("=> init deconv weights from normal distribution")
        for m in self.deconv_layers.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        logger.info("=> init final conv weights from normal distribution")
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def view_to_bcdhw(self, x: torch.Tensor) -> torch.Tensor:
        """
        view a falttened 2D heatmap to 3D heatmap, sharing the same memory by using view()
        Args:
            x: TENSOR (BATCH, NCLASSES * DEPTH, HEIGHT|ROWS, WIDTH|COLS)

        Returns:
            TENSOR (BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH)
        """
        return x.contiguous().view(
            x.shape[0],  # BATCH,
            self.nclasses,  # NCLASSES
            self.depth_res,  # DEPTH
            self.height_res,  # HEIGHT,
            self.width_res,  # WIDTH
        )

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError()

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), "ERROR: num_deconv_layers is different len(num_deconv_filters)"
        assert num_layers == len(num_kernels), "ERROR: num_deconv_layers is different len(num_deconv_filters)"

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias,
                ))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        x = kwargs["feature"]
        x = self.deconv_layers(x)
        x = self.final_layer(x)

        x = x.reshape((x.shape[0], self.nclasses, -1))  # TENSOR (B, NCLASS, DEPTH x HEIGHT x WIDTH)
        x = norm_heatmap(self.norm_type, x)  # TENSOR (B, NCLASS, DEPTH x HEIGHT x WIDTH)

        confi = torch.max(x, dim=-1).values  # TENSOR (B, NCLASS)
        assert x.dim() == 3, f"Unexpected dim, expect x has shape (B, C, DxHxW), got {x.shape}"
        x = x / (x.sum(dim=-1, keepdim=True) + 1e-7)
        x = self.view_to_bcdhw(x)  # TENSOR(BATCH, NCLASSES, DEPTH, HEIGHT, WIDTH)
        x = integral_heatmap3d(x)  # TENSOR (BATCH, NCLASSES, 3)
        return {"uvd": x, "uvd_confi": confi}

