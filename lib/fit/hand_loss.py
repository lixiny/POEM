import numpy as np
import torch
from manotorch.utils.quatutils import quaternion_norm_squared, quaternion_inv, quaternion_mul


class HandLoss:

    @staticmethod
    def pose_quat_norm_loss(var_pose):
        """this is the only loss accepts unnormalized quats"""
        reshaped_var_pose = var_pose.reshape((-1, 16, 4))  # TENSOR[B, 16, 4]
        quat_norm_sq = quaternion_norm_squared(reshaped_var_pose)  # TENSOR[B, 16 ]
        squared_norm_diff = quat_norm_sq - 1.0  # TENSOR[B, 16]
        res = torch.mean(torch.pow(squared_norm_diff, 2))
        return res

    @staticmethod
    def pose_reg_loss(var_pose_normed, var_pose_init):
        # the format of quat is [w, x, y, z]
        # to regularize
        # just to make sure w is close to 1.0
        # working aside with self.pose_quat_norm_loss defined above
        inv_var_pose_init = quaternion_inv(var_pose_init)  # (B, 15, 4)
        combined_pose = quaternion_mul(var_pose_normed, inv_var_pose_init)  # (B, 15, 4)
        w = combined_pose[..., 0]  # get w
        diff = w - 1.0  # TENSOR[B, 15]
        res = torch.mean(torch.pow(diff, 2))
        return res

    @staticmethod
    def shape_reg_loss(var_shape, shape_init):
        return torch.mean(torch.sum(torch.pow(var_shape - shape_init, 2), dim=-1))

    # **** axis order right hand

    #         14-13-12-\
    #                   \
    #    2-- 1 -- 0 -----*
    #   5 -- 4 -- 3 ----/
    #   11 - 10 - 9 ---/
    #    8-- 7 -- 6 --/

    @staticmethod
    def joint_b_axis_loss(b_axis, axis, angle_mask):
        # b_axis [B, 15, 3]
        b_soft_idx = [0, 3, 9, 6]
        b_thumb_soft_idx = [12]
        b_restrict_idx = [i for i in range(15) if i not in b_soft_idx and i not in b_thumb_soft_idx]

        b_axis_cos = torch.einsum("bki,bki->bk", b_axis, axis)
        # print(torch.sum(b_axis).item(), torch.sum(axis).item(), torch.sum(b_axis_cos).item())
        restrict_cos = b_axis_cos[:, b_restrict_idx]
        soft_loss = torch.relu(torch.abs(b_axis_cos[:, b_soft_idx]) - np.cos(np.pi / 2 - np.pi / 36))  # [-5, 5]
        thumb_soft_loss = torch.relu(torch.abs(b_axis_cos[:, b_thumb_soft_idx]) -
                                     np.cos(np.pi / 2 - np.pi / 9))  # [-60, 60]

        res = (torch.mean(torch.pow(restrict_cos * angle_mask[:, b_restrict_idx], 2)) +
               torch.mean(torch.pow(soft_loss * angle_mask[:, b_soft_idx], 2)) +
               torch.mean(torch.pow(thumb_soft_loss * angle_mask[:, b_thumb_soft_idx], 2)))
        return res

    @staticmethod
    def joint_u_axis_loss(u_axis, axis, angle_mask):
        # u_axis [B, 15, 3]
        u_soft_idx = [0, 3, 9, 6]
        u_thumb_soft_idx = [12]
        u_restric_idx = [i for i in range(15) if i not in u_soft_idx and i not in u_thumb_soft_idx]

        u_axis_cos = torch.einsum("bki,bki->bk", u_axis, axis)
        restrict_cos = u_axis_cos[:, u_restric_idx]
        soft_loss = torch.relu(torch.abs(u_axis_cos[:, u_soft_idx]) - np.cos(np.pi / 2 - np.pi / 6))  # [-10, 10]
        thumb_soft_loss = torch.relu(torch.abs(u_axis_cos[:, u_thumb_soft_idx]) -
                                     np.cos(np.pi / 2 - np.pi / 3))  # [-60, 60]

        res = (torch.mean(torch.pow(restrict_cos * angle_mask[:, u_restric_idx], 2)) +
               torch.mean(torch.pow(soft_loss * angle_mask[:, u_soft_idx], 2)) +
               torch.mean(torch.pow(thumb_soft_loss * angle_mask[:, u_thumb_soft_idx], 2)))
        return res

    @staticmethod
    def joint_l_limit_loss(l_axis, axis, angle_mask):
        # l_axis [B, 15, 3]
        l_soft_idx = [0, 3, 9, 6]
        l_thumb_soft_idx = [12]
        l_restrict_idx = [i for i in range(15) if i not in l_soft_idx and i not in l_thumb_soft_idx]

        l_axis_cos = torch.einsum("bki,bki->bk", l_axis, axis)
        restrict_cos = l_axis_cos[:, l_restrict_idx]
        soft_loss = torch.relu(-l_axis_cos[:, l_soft_idx] + 1 - np.cos(np.pi / 2 - np.pi / 9))  # [-20, 20]
        thumb_soft_loss = torch.relu(-l_axis_cos[:, l_thumb_soft_idx] + 1 - np.cos(np.pi / 2 - np.pi / 3))

        res = (torch.mean(torch.pow((restrict_cos - 1) * angle_mask[:, l_restrict_idx], 2)) +
               torch.mean(torch.pow(soft_loss * angle_mask[:, l_soft_idx], 2)) +
               torch.mean(torch.pow(thumb_soft_loss * angle_mask[:, l_thumb_soft_idx], 2)))
        return res

    @staticmethod
    def rotation_angle_loss(angle, limit_angle=np.pi / 2, eps=1e-10):
        angle_new = torch.zeros_like(angle)  # TENSOR[B, 15]
        nonzero_mask = torch.abs(angle) > eps  # TENSOR[B, 15], bool
        angle_new[nonzero_mask] = angle[nonzero_mask]  # if angle is too small, pick them out of backward graph
        angle_over_limit = torch.relu(angle_new - limit_angle)  # < pi/2, 0; > pi/2, linear | Tensor[16, ]
        angle_over_limit_squared = torch.pow(angle_over_limit, 2)  # TENSOR[15, ]
        res = torch.mean(angle_over_limit_squared)
        return res
