import os

import cv2
import numpy as np
import torch
from lib.datasets import DexYCB, DexYCBMultiView
from lib.utils.builder import DATASET
from lib.utils.config import CN
from lib.utils.logger import logger
from termcolor import cprint
from torch.utils.data._utils.collate import default_collate


def map2uv(map, size=(224, 224)):
    if map.ndim == 4:
        uv = np.zeros((map.shape[0], map.shape[1], 2))
        uv_conf = np.zeros((map.shape[0], map.shape[1], 1))
        map_size = map.shape[2:]
        for j in range(map.shape[0]):
            for i in range(map.shape[1]):
                uv_conf[j][i] = map[j, i].max()
                max_pos = map[j, i].argmax()
                uv[j][i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
                uv[j][i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]
    else:
        uv = np.zeros((map.shape[0], 2))
        uv_conf = np.zeros((map.shape[0], 1))
        map_size = map.shape[1:]
        for i in range(map.shape[0]):
            uv_conf[i] = map[i].max()
            max_pos = map[i].argmax()
            uv[i][1] = (max_pos // map_size[1]) / map_size[0] * size[0]
            uv[i][0] = (max_pos % map_size[1]) / map_size[1] * size[1]

    return uv, uv_conf


def uv2map(uv, size=(224, 224)):
    kernel_size = (size[0] * 13 // size[0] - 1) // 2
    gaussian_map = np.zeros((uv.shape[0], size[0], size[1]))
    size_transpose = np.array(size)
    gaussian_kernel = cv2.getGaussianKernel(2 * kernel_size + 1, (2 * kernel_size + 2) / 4.)
    gaussian_kernel = np.dot(gaussian_kernel, gaussian_kernel.T)
    gaussian_kernel = gaussian_kernel / gaussian_kernel.max()

    for i in range(gaussian_map.shape[0]):
        if (uv[i] >= 0).prod() == 1 and (uv[i][1] <= size_transpose[0]) and (uv[i][0] <= size_transpose[1]):
            s_pt = np.array((uv[i][1], uv[i][0]))
            p_start = s_pt - kernel_size
            p_end = s_pt + kernel_size
            p_start_fix = (p_start >= 0) * p_start + (p_start < 0) * 0
            k_start_fix = (p_start >= 0) * 0 + (p_start < 0) * (-p_start)
            p_end_fix = (p_end <= (size_transpose - 1)) * p_end + (p_end > (size_transpose - 1)) * (size_transpose - 1)
            k_end_fix = (p_end <= (size_transpose - 1)) * kernel_size * 2 + \
                (p_end > (size_transpose - 1)) * (2*kernel_size - (p_end - (size_transpose - 1)))

            gaussian_map[i, p_start_fix[0]: p_end_fix[0] + 1, p_start_fix[1]: p_end_fix[1] + 1] = \
                gaussian_kernel[k_start_fix[0]: k_end_fix[0] + 1, k_start_fix[1]: k_end_fix[1] + 1]

    return gaussian_map


class CMRDataConverter(object):

    def __init__(self, center_idx, is_train=True):

        from .net import Pool
        self.Pool = Pool
        self.center_idx = center_idx
        self.V_STD = 0.2
        self.is_train = is_train

        self.has_spiral_transform = False
        self.spiral_indices_list = []
        self.down_sample_list = []
        self.up_transform_list = []
        self.faces = []

    def convert(self, inputs):
        """
        Convert the data to the format that the CMR can accept.
        """
        if not self.has_spiral_transform:
            # The function: spiral_transform() must be called at runtime to get the down_transform_list.
            # Otherwise, (if you are using torch DDP) you will get ``RuntimeError: sparse tensors do not have storage''
            # This is because sparse tensors on stable pytorch do not support pickling.
            # One workaround at least until pytorch 1.8.2 LTS is simply to construct to SparseTensor within
            # your getitem function rather than to scope it in. For example:
            from .utils import spiral_tramsform
            work_dir = os.path.dirname(os.path.realpath(__file__))
            transf_pth = os.path.join(work_dir, 'template', 'transform.pkl')
            template_pth = os.path.join(work_dir, 'template', 'template.ply')
            spiral_indices_list, down_sample_list, up_transform_list, tmp = spiral_tramsform(transf_pth, template_pth)

            self.spiral_indices_list = spiral_indices_list
            self.down_sample_list = down_sample_list
            self.up_transform_list = up_transform_list
            self.faces = tmp['face']
            self.has_spiral_transform = True

        img = inputs["image"]  # (C, H, W)
        mask = inputs["mask"]  # (C, H, W)
        v0 = inputs["target_verts_3d"][:778]

        K = inputs["target_cam_intr"]  # (3, 3)
        xyz = inputs["target_joints_3d"]  # (21, 3)
        uv = inputs["target_joints_2d"]  # (21, 2)

        uv_map = uv2map(uv.astype(np.int), img.shape[1:]).astype(np.float32)  # (21, H, W)
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2] // 2, img.shape[1] // 2)).transpose(2, 0, 1)
        mask = cv2.resize(mask, (img.shape[2] // 2, img.shape[1] // 2))

        xyz_root = xyz[self.center_idx]  # (3)
        v0 = (v0 - xyz_root) / self.V_STD
        xyz = (xyz - xyz_root) / self.V_STD

        v0 = torch.from_numpy(v0).float()
        if self.is_train:
            v1 = self.Pool(v0.unsqueeze(0), self.down_sample_list[0])[0]
            v2 = self.Pool(v1.unsqueeze(0), self.down_sample_list[1])[0]
            v3 = self.Pool(v2.unsqueeze(0), self.down_sample_list[2])[0]
            gt = [v0, v1, v2, v3]
        else:
            gt = [v0]

        data = {
            'img': img,
            'mesh_gt': gt,
            'K': K,
            'mask_gt': mask,
            'xyz_gt': xyz,
            'uv_point': uv,
            'uv_gt': uv_map,
            'xyz_root': xyz_root
        }

        return data


@DATASET.register_module()
class DexYCB_CMR(DexYCB):

    def __init__(self, cfg: CN):
        super(DexYCB_CMR, self).__init__(cfg)
        is_train = ("train" in self.data_split)
        self.CMR_DC = CMRDataConverter(self.center_idx, is_train)
        logger.warning(f"Initialized child class: DexYCB_CMR (DexYCB)")

    def getitem_3d(self, idx):
        res = super().getitem_3d(idx)
        res = self.CMR_DC.convert(res)
        return res


@DATASET.register_module()
class DexYCBMultiView_CMR(DexYCBMultiView):

    def __init__(self, cfg: CN):
        super(DexYCBMultiView_CMR, self).__init__(cfg)
        is_train = ("train" in self.data_split)
        self.CMR_DC = CMRDataConverter(self.center_idx, is_train)
        logger.warning(f"Initialized child class: DexYCBMultiView_CMR (DexYCBMultiView)")

    def __getitem__(self, idx):
        res = super().__getitem__(idx)

        nv_img = res["image"]  # (NPERSP, C, H, W)
        nv_mask = res["mask"]  # (NPERSP, C, H, W)
        nv_target_verts_3d = res["target_verts_3d"]  # (NPERSP, 778, 3)
        nv_target_cam_intr = res["target_cam_intr"]  # (NPERSP, 3, 3)
        nv_target_cam_extr = res["target_cam_extr"]  # (NPERSP, 4, 4)
        nv_target_joints_3d = res["target_joints_3d"]  # (NPERSP, 21, 3)
        nv_target_joints_2d = res["target_joints_2d"]  # (NPERSP, 21, 2)

        all_res = []

        for i in range(self.n_views):
            curr_res = {
                "image": nv_img[i],
                "mask": nv_mask[i],
                "target_verts_3d": nv_target_verts_3d[i],
                "target_cam_intr": nv_target_cam_intr[i],
                "target_joints_3d": nv_target_joints_3d[i],
                "target_joints_2d": nv_target_joints_2d[i],
            }
            curr_res = self.CMR_DC.convert(curr_res)
            curr_res["extr"] = np.linalg.inv(nv_target_cam_extr[i])
            all_res.append(curr_res)

        all_res = default_collate(all_res)
        return all_res
