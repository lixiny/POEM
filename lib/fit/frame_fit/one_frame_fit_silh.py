import numpy as np
import torch
from lib.utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection)
from manotorch.axislayer import AxisLayer
from manotorch.manolayer import ManoLayer, MANOOutput
from manotorch.utils.quatutils import (normalize_quaternion, quaternion_to_angle_axis)

from ..hand_loss import HandLoss
from ..silhouette_loss import MultiviewSilhouetteLoss


class OneFrameFitSilh:

    def __init__(
        self,
        device: torch.device,
        mano_layer: ManoLayer,
        lr: float = 1e-2,
        grad_clip: float = None,
        side: str = "right",
        image_size: int = 256,
        silh_size: int = 128,
        lambda_reprojection_loss: float = 1000.0,
        lambda_anatomical_loss: float = 5.0,
        lambda_mask_render_loss: float = 0.0,
        gamma_joint_b_axis_loss: float = 1.0,
        gamma_joint_u_axis_loss: float = 1.0,
        gamma_joint_l_limit_loss: float = 0.01,
        gamma_angle_limit_loss: float = 0.00,
    ) -> None:
        self.device = device
        self.lr = lr
        self.side = side
        self.grad_clip = grad_clip

        # mano layer
        self.mano_layer = mano_layer
        self.axis_layer = AxisLayer().to(self.device)
        self.silh_loss = MultiviewSilhouetteLoss(self.device, self.mano_layer, img_size=image_size,
                                                 silh_size=silh_size).to(self.device)

        # n_iter
        self._n_iter: int = 0

        # all values
        self._image_scale: int = None

        self._hand_pose: torch.Tensor = None
        self._hand_shape: torch.Tensor = None
        self._hand_tsl: torch.Tensor = None
        self._fit_hand_pose: bool = False
        self._fit_hand_shape: bool = False
        self._fit_hand_tsl: bool = False
        self._keypoint_arr: torch.Tensor = None
        self._keypoint_weight_arr: torch.Tensor = None
        self._cam_intr_arr: torch.Tensor = None
        self._cam_extr_arr: torch.Tensor = None
        self._n_persp: int = None

        self._enable_silh_loss = None

        self.lambda_reprojection_loss = lambda_reprojection_loss
        self.lambda_anatomical_loss = lambda_anatomical_loss
        self.lambda_mask_render_loss = lambda_mask_render_loss

        self.gamma_joint_b_axis_loss = gamma_joint_b_axis_loss
        self.gamma_joint_u_axis_loss = gamma_joint_u_axis_loss
        self.gamma_joint_l_limit_loss = gamma_joint_l_limit_loss
        self.gamma_angle_limit_loss = gamma_angle_limit_loss

        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None
        self.optimizing = False
        self.curr_iter = -1

    def setup(
        self,
        n_iter: int,
        image_scale: int,
        enable_silh_loss: bool,
        const_keypoint_arr,
        const_mask_arr,
        const_cam_intr_arr,
        const_cam_extr_arr,
        init_hand_pose,
        init_hand_shape,
        init_hand_tsl,
        fit_hand_pose=True,
        fit_hand_shape=True,
        fit_hand_tsl=True,
    ):
        self._n_iter = n_iter
        self._image_scale = image_scale

        self._keypoint_arr = const_keypoint_arr[..., 0:2].clone().detach().float().to(self.device)  # [NPERSP, 21, 2]
        self._keypoint_weight_arr = const_keypoint_arr[..., 2].clone().detach().float().to(self.device)  # [NPERSP, 21]
        self._mask_arr = const_mask_arr.clone().detach().float().to(self.device)  # [NPERSP, H, W]
        self._cam_intr_arr = const_cam_intr_arr.clone().detach().float().to(self.device)  # [NPERSP, 3, 3]
        self._cam_extr_arr = const_cam_extr_arr.clone().detach().float().to(self.device)  # [NPERSP, 4, 4]
        self._n_persp = int(self._keypoint_arr.shape[0])

        self._hand_pose = init_hand_pose.float().detach().clone().to(self.device)  # (1, 64)
        self._hand_shape = init_hand_shape.float().detach().clone().to(self.device)  # (1, 10)
        self._hand_tsl = init_hand_tsl.float().detach().clone().to(self.device)  # (1, 3)

        self._fit_hand_pose = fit_hand_pose
        self._fit_hand_shape = fit_hand_shape
        self._fit_hand_tsl = fit_hand_tsl

        self._enable_silh_loss = enable_silh_loss

        param_to_optimize = []
        if fit_hand_pose:
            self._hand_pose.requires_grad_(True)
            param_to_optimize.append({"params": [self._hand_pose]})
        if fit_hand_shape:
            self._hand_shape.requires_grad_(True)
            param_to_optimize.append({"params": [self._hand_shape], "lr": 0.1 * self.lr})
        if fit_hand_tsl:
            self._hand_tsl.requires_grad_(True)
            param_to_optimize.append({"params": [self._hand_tsl]})

        # optimizer
        if len(param_to_optimize) > 0:
            self.optimizer = torch.optim.Adam(param_to_optimize, lr=self.lr)
            # self.optimizer = torch.optim.AdamW(param_to_optimize, lr=self.lr)
            self.optimizing = True
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                min_lr=1e-5,
                mode="min",
                factor=0.5,
                threshold=1e-4,
                threshold_mode="rel",
                patience=50,
                verbose=False,
            )
        else:
            self.optimizing = False

        self.curr_iter = 0

    def reset(self):
        self._n_iter = 0
        self._image_scale = None

        self._hand_pose: torch.Tensor = None
        self._hand_shape: torch.Tensor = None
        self._hand_tsl: torch.Tensor = None
        self._fit_hand_pose: bool = False
        self._fit_hand_shape: bool = False
        self._fit_hand_tsl: bool = False
        self._keypoint_arr: torch.Tensor = None
        self._keypoint_weight_arr: torch.Tensor = None
        self._mask_arr: torch.Tensor = None
        self._cam_intr_arr: torch.Tensor = None
        self._cam_extr_arr: torch.Tensor = None
        self._n_persp: int = None

        self._enable_silh_loss = None

        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = None
        self.optimizing = False
        self.curr_iter = -1

    def step(self):
        bsize = self._hand_pose.shape[0]  # 1
        vec_pose = self._hand_pose  # [1, 64]
        vec_shape = self._hand_shape  # [1, 10]
        vec_tsl = self._hand_tsl  # [1, 3]

        # region ====== hand anatomical loss >>>>>>
        pose_quat_norm_loss = HandLoss.pose_quat_norm_loss(vec_pose)
        vec_pose_normalized = normalize_quaternion(vec_pose.reshape((-1, 16, 4)))  # (B=1, 16, 4)
        n_var_pose = 15
        init_pose_zero = np.array([[1.0, 0.0, 0.0, 0.0]] * n_var_pose)[None, ...].astype(np.float32)  # (B=1, 15, 4)
        init_pose_zero = torch.tensor(init_pose_zero, dtype=torch.float, device=self.device).expand((bsize, -1, -1))
        pose_reg_loss = HandLoss.pose_reg_loss(vec_pose_normalized[:, 1:, :], init_pose_zero)
        init_shape_zero = torch.zeros((1, 10), dtype=torch.float, device=self.device).expand((bsize, -1))  # (B=1, 10)
        shape_reg_loss = HandLoss.shape_reg_loss(vec_shape, init_shape_zero)

        vec_pose_normalized_flat = vec_pose_normalized.reshape((bsize, -1))  # (B=1, 64)
        mano_out: MANOOutput = self.mano_layer(vec_pose_normalized_flat, vec_shape)

        rebuild_joints = mano_out.joints  # [B=1, 21, 3]
        rebuild_verts = mano_out.verts
        rebuild_transforms = mano_out.transforms_abs
        rebuild_joints = rebuild_joints + vec_tsl.unsqueeze(1)  # [B=1, 21, 3]
        rebuild_verts = rebuild_verts + vec_tsl.unsqueeze(1)

        b_axis, u_axis, l_axis = self.axis_layer(rebuild_joints, rebuild_transforms)  # (B, 15, 3)
        angle_axis = quaternion_to_angle_axis(vec_pose_normalized)  # (B, 16, 3)
        angle_axis = angle_axis[:, 1:, :]  # ignore global rot [B, 15, 3]
        axis = angle_axis / (torch.norm(angle_axis, dim=2, keepdim=True) + 1e-9)
        angle = torch.norm(angle_axis, dim=2, keepdim=False)
        angle_mask = (angle >= 1e-2).float()

        # limit angle
        angle_limit_loss = HandLoss.rotation_angle_loss(angle)
        joint_b_axis_loss = HandLoss.joint_b_axis_loss(b_axis, axis, angle_mask)
        joint_u_axis_loss = HandLoss.joint_u_axis_loss(u_axis, axis, angle_mask)
        joint_l_limit_loss = HandLoss.joint_l_limit_loss(l_axis, axis, angle_mask)

        hand_anatomical_loss = (1.0 * pose_quat_norm_loss + 0.0 * pose_reg_loss + 0.1 * shape_reg_loss +
                                self.gamma_angle_limit_loss * angle_limit_loss +
                                self.gamma_joint_b_axis_loss * joint_b_axis_loss +
                                self.gamma_joint_u_axis_loss * joint_u_axis_loss +
                                self.gamma_joint_l_limit_loss * joint_l_limit_loss)
        # endregion <<<<<<

        # region ====== reprojection loss >>>>>>
        # transform to 2d
        multi_cam_rebuild_joints = batch_cam_extr_transf(
            self._cam_extr_arr[None, ...].expand(bsize, -1, -1, -1),  # (B=1, NPERSP, 4, 4)
            rebuild_joints.unsqueeze(1).expand(-1, self._n_persp, -1, -1),  # (B=1, NPERSP, 21, 3)
        )  # [B, NPERSP, 21, 3]
        multi_cam_projected_joints = batch_cam_intr_projection(
            self._cam_intr_arr[None, ...].expand(bsize, -1, -1, -1),  # (B, NPERSP, 3, 3)
            multi_cam_rebuild_joints,  # [B, NPERSP, 21, 3]
        )  # [B, NPERSP, 21, 2]

        multi_cam_projected_offset = (multi_cam_projected_joints - self._keypoint_arr) / self._image_scale
        multi_cam_projected_distance = torch.sum(torch.pow(multi_cam_projected_offset, 2), dim=3)  # [B, NPERSP, 21]

        multi_cam_projected_distance = multi_cam_projected_distance * self._keypoint_weight_arr
        reprojection_loss = torch.mean(multi_cam_projected_distance)
        # endregion <<<<<<

        # region ====== silh loss >>>>>>
        if self._enable_silh_loss:
            mask_render_loss = self.silh_loss(
                self._cam_intr_arr,
                self._cam_extr_arr,
                rebuild_verts.squeeze(0),
                self._mask_arr,
            )
        else:
            mask_render_loss = 0.0

        loss = (self.lambda_reprojection_loss * reprojection_loss + self.lambda_mask_render_loss * mask_render_loss +
                self.lambda_anatomical_loss * hand_anatomical_loss)

        # step
        if self.optimizing:
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([self._hand_pose, self._hand_shape, self._hand_tsl], self.grad_clip)
            self.optimizer.step()
            self.scheduler.step(loss)

        self.curr_iter += 1
        return reprojection_loss.item(), mask_render_loss.item()

    def condition(self):
        return self.curr_iter < self._n_iter

    def get_total_iter(self):
        return self._n_iter

    def recover_hand(self, squeeze_out=True):
        vec_pose = self._hand_pose.detach()
        vec_shape = self._hand_shape.detach()
        vec_tsl = self._hand_tsl.detach()

        mano_out: MANOOutput = self.mano_layer(vec_pose, vec_shape)
        rebuild_joints = mano_out.joints  # [B, 21, 3]
        rebuild_verts = mano_out.verts
        rebuild_joints = rebuild_joints + vec_tsl.unsqueeze(1)  # [B,21,3]
        rebuild_verts = rebuild_verts + vec_tsl.unsqueeze(1)  # [B,NVERT,3]

        if squeeze_out:
            if rebuild_joints.shape[0] != 1:
                raise RuntimeError("cannot squeeze output due to batch dim")
            rebuild_joints = rebuild_joints.squeeze(0)
            rebuild_verts = rebuild_verts.squeeze(0)

        return rebuild_joints, rebuild_verts

    def optimized_value(self, squeeze_out=False):
        hand_pose = self._hand_pose.detach().clone().cpu()
        hand_shape = self._hand_shape.detach().clone().cpu()
        hand_tsl = self._hand_tsl.detach().clone().cpu()
        if squeeze_out:
            if hand_pose.shape[0] != 1:
                raise RuntimeError("cannot squeeze output due to batch dim")
            hand_pose = hand_pose.squeeze(0)
            hand_shape = hand_shape.squeeze(0)
            hand_tsl = hand_tsl.squeeze(0)
        return {
            "hand_pose": hand_pose,
            "hand_shape": hand_shape,
            "hand_tsl": hand_tsl,
        }

    def get_rest_joints(self):
        init_hand_pose = torch.zeros(16, 4, device=self.device)
        init_hand_pose[:, 0] = 1.0
        init_hand_pose = init_hand_pose.reshape((1, 64))
        init_hand_shape = torch.zeros(1, 10, device=self.device)  # [1, 10]
        mano_out: MANOOutput = self.mano_layer(init_hand_pose, init_hand_shape)
        rebuild_joints = mano_out.joints  # [B, 21, 3]
        return rebuild_joints.squeeze(0)
