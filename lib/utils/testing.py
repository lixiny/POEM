import os
import pickle

import numpy as np
import torch
from manotorch.manolayer import ManoLayer

from lib.metrics.pck import Joint3DPCK, Vert3DPCK
from lib.utils.transform import (batch_cam_extr_transf, batch_cam_intr_projection, denormalize)
from lib.viztools.draw import save_a_image_with_mesh_joints
from lib.viztools.opendr_renderer import OpenDRRenderer

from .logger import logger


class IdleCallback():

    def __init__(self):
        pass

    def __call__(self, preds, inputs, step_idx, **kwargs):
        pass

    def on_finished(self):
        pass

    def reset(self):
        pass


class AUCCallback(IdleCallback):

    def __init__(self, exp_dir, val_min=0.0, val_max=0.02, steps=20):
        self.exp_dir = exp_dir
        self.val_min = val_min
        self.val_max = val_max
        self.steps = steps
        self.PCK_J = Joint3DPCK(EVAL_TYPE="joints_3d", VAL_MIN=val_min, VAL_MAX=val_max, STEPS=steps)
        self.PCK_V = Vert3DPCK(EVAL_TYPE="verts_3d", VAL_MIN=val_min, VAL_MAX=val_max, STEPS=steps)

    def reset(self):
        self.PCK_J.reset()
        self.PCK_V.reset()

    def __call__(self, preds, inputs, step_idx, **kwargs):
        self.PCK_J.feed(preds, inputs)
        self.PCK_V.feed(preds, inputs)

    def on_finished(self):

        logger.info(f"Dump AUC results to {self.exp_dir}")
        filepth_j = os.path.join(self.exp_dir, 'res_auc_j.pkl')
        auc_pth_j = os.path.join(self.exp_dir, 'auc_j.txt')
        filepth_v = os.path.join(self.exp_dir, 'res_auc_v.pkl')
        auc_pth_v = os.path.join(self.exp_dir, 'auc_v.txt')

        dict_J = self.PCK_J.get_measures()
        dict_V = self.PCK_V.get_measures()

        with open(filepth_j, 'wb') as f:
            pickle.dump(dict_J, f)
        with open(auc_pth_j, 'w') as ff:
            ff.write(str(dict_J["auc_all"]))

        with open(filepth_v, 'wb') as f:
            pickle.dump(dict_V, f)
        with open(auc_pth_v, 'w') as ff:
            ff.write(str(dict_V["auc_all"]))

        logger.warning(f"auc_j: {dict_J['auc_all']}")
        logger.warning(f"auc_v: {dict_V['auc_all']}")
        self.reset()


class DrawingHandCallback(IdleCallback):

    def __init__(self, img_draw_dir):

        self.img_draw_dir = img_draw_dir
        os.makedirs(img_draw_dir, exist_ok=True)

        mano_layer = ManoLayer(mano_assets_root="assets/mano_v1_2")
        self.mano_faces = mano_layer.get_mano_closed_faces().numpy()
        self.renderer = OpenDRRenderer()

    def __call__(self, preds, inputs, step_idx, **kwargs):

        tensor_image = inputs["image"]  # (B, N, 3, H, W) 5 channels
        batch_size = tensor_image.size(0)
        n_views = tensor_image.size(1)
        image = denormalize(tensor_image, [0.5, 0.5, 0.5], [1, 1, 1], inplace=False)
        image = image.permute(0, 1, 3, 4, 2)
        image = image.mul_(255.0).detach().cpu()  # (B, N, H, W, 3)
        image = image.numpy().astype(np.uint8)

        cam_param = inputs["target_cam_intr"]
        mesh_xyz = preds["pred_verts_3d"].unsqueeze(1).repeat(1, n_views, 1, 1)
        pose_xyz = preds["pred_joints_3d"].unsqueeze(1).repeat(1, n_views, 1, 1)

        gt_T_c2m = torch.linalg.inv(inputs["target_cam_extr"])  # (B, N, 4, 4)
        mesh_xyz = batch_cam_extr_transf(gt_T_c2m, mesh_xyz)  # (B, N, 21, 3)
        pose_xyz = batch_cam_extr_transf(gt_T_c2m, pose_xyz)  # (B, N, 778, 3)
        pose_uv = batch_cam_intr_projection(cam_param, pose_xyz)  # (B, N, 21, 2)

        mesh_xyz = mesh_xyz.detach().cpu().numpy()
        pose_xyz = pose_xyz.detach().cpu().numpy()
        pose_uv = pose_uv.detach().cpu().numpy()
        cam_param = cam_param.detach().cpu().numpy()

        for i in range(batch_size):
            for j in range(n_views):
                file_name = os.path.join(self.img_draw_dir, f"step{step_idx}_frame{i}_view{j}.jpg")
                save_a_image_with_mesh_joints(image=image[i, j],
                                              cam_param=cam_param[i, j],
                                              mesh_xyz=mesh_xyz[i, j],
                                              pose_uv=pose_uv[i, j],
                                              pose_xyz=pose_xyz[i, j],
                                              face=self.mano_faces,
                                              with_mayavi_mesh=False,
                                              with_skeleton_3d=False,
                                              file_name=file_name,
                                              renderer=self.renderer)

    def on_finished(self):
        pass
