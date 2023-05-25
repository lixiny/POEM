from typing import Optional

import torch
import torch.nn as nn
from manotorch.manolayer import ManoLayer, MANOOutput

# https://github.com/lmb-freiburg/freihand/blob/master/utils/mano_utils.py
kpId2vertices = {
    4: [744],  #ThumbT
    8: [320],  #IndexT
    12: [443],  #MiddleT
    16: [555],  #RingT
    20: [672]  #PinkT
}


class MANO(nn.Module):

    def __init__(self, **kwargs):
        super(MANO, self).__init__()
        self.center_idx = kwargs["center_idx"]
        self.mano_layer = ManoLayer(**kwargs)

    def forward(self, pose_coeffs, betas=None, **kwargs):
        mano_output: MANOOutput = self.mano_layer(pose_coeffs, betas, **kwargs)

        verts_xyz = mano_output.verts
        joints_reg = torch.matmul(self.mano_layer.th_J_regressor, verts_xyz)
        tipsId = [v[0] for k, v in kpId2vertices.items()]
        tips = verts_xyz[:, tipsId]
        joints_reg = torch.cat([joints_reg, tips], dim=1)
        # Reorder joints to match OpenPose definition
        joints_reg = joints_reg[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]

        if self.center_idx is not None:
            center_joint_reg = joints_reg[:, self.center_idx].unsqueeze(1)
        else:  # dummy center joint (B, 1, 3)
            center_joint_reg = torch.zeros_like(joints_reg[:, 0].unsqueeze(1))

        verts_rel = verts_xyz - center_joint_reg
        joints_reg_rel = joints_reg - center_joint_reg

        rot_abs = mano_output.transforms_abs[:, :, :3, :3]
        results = {
            "_joints_rel": mano_output.joints,
            "_verts_rel": mano_output.verts,
            "verts_rel": verts_rel,
            "joints_reg_rel": joints_reg_rel,
            "center_idx": mano_output.center_idx,
            "center_joints_reg": center_joint_reg,
            "full_poses": mano_output.full_poses,
            "betas": mano_output.betas,
            "rot_abs": rot_abs,
        }
        return results
