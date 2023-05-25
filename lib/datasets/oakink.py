import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import List
import warnings

import imageio
import numpy as np
import torch
from manotorch.manolayer import ManoLayer
from termcolor import colored

from ..utils.builder import DATASET
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import get_annot_center, get_annot_scale, persp_project
from ..utils.transform import quat_to_rotmat, rotmat_to_aa, quat_to_aa
from .hdata import HDataset, kpId2vertices


@DATASET.register_module()
class OakInk(HDataset):

    @staticmethod
    def _get_info_list(data_dir, use_split_mv, split_key, data_split):
        if use_split_mv:
            mid_key = "anno_mv"
        else:
            mid_key = "anno"

        if data_split == "train+val":
            info_list = json.load(open(os.path.join(data_dir, mid_key, "split", split_key, "seq_train.json")))
        elif data_split == "train":
            info_list = json.load(
                open(os.path.join(data_dir, mid_key, "split_train_val", split_key, "example_split_train.json")))
        elif data_split == "val":
            info_list = json.load(
                open(os.path.join(data_dir, mid_key, "split_train_val", split_key, "example_split_val.json")))
        else:  # data_split == "test":
            info_list = json.load(open(os.path.join(data_dir, mid_key, "split", split_key, "seq_test.json")))
        return info_list

    def __init__(self, cfg):
        super(OakInk, self).__init__(cfg)
        self.rMANO = ManoLayer(side="right", mano_assets_root="assets/mano_v1_2")
        assert self.data_mode in ["2D", "UVD", "3D"], f"OakInk does not dupport {self.data_mode} mode"
        assert self.data_split in [
            "all",
            "train+val",
            "train",
            "val",
            "test",
        ], "OakInk data_split must be one of ['train', 'val']"
        self.split_mode = cfg.SPLIT_MODE
        assert self.split_mode in [
            "default",
            "subject",
            "object",
        ], "OakInk split_mode must be one of ['default', 'subject', 'object]"
        self.use_split_mv = cfg.USE_SPLIT_MV
        if self.use_split_mv:
            self.packed_key = "anno_packed_mv"
        else:
            self.packed_key = "anno_packed"
        self.use_pack = cfg.get("USE_PACK", False)
        if self.use_pack:
            self.getitem_3d = self._getitem_3d_pack
        else:
            self.getitem_3d = self._getitem_3d

        self.load_dataset()
        logger.info(f"initialized child class: {self.name}")

    def load_dataset(self):
        self.name = "oakink-image"
        self.root = os.path.join(self.data_root, "OakInk", "image")

        if self.data_split == "all":
            self.info_list = json.load(open(os.path.join(self.root, "anno", "seq_all.json")))
        elif self.split_mode == "default":
            self.info_list = self._get_info_list(self.root, self.use_split_mv, "split0", self.data_split)
        elif self.split_mode == "subject":
            self.info_list = self._get_info_list(self.root, self.use_split_mv, "split1", self.data_split)
        else:  # self.split_mode == "object":
            self.info_list = self._get_info_list(self.root, self.use_split_mv, "split2", self.data_split)

        self.info_str_list = []
        for info in self.info_list:
            info_str = "__".join([str(x) for x in info])
            info_str = info_str.replace("/", "__")
            self.info_str_list.append(info_str)

        self.framedata_color_name = [
            "north_east_color",
            "south_east_color",
            "north_west_color",
            "south_west_color",
        ]

        self.n_samples = len(self.info_str_list)
        self.sample_idxs = list(range(self.n_samples))
        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{self.n_samples} samples for data_split {self.data_split}")

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idx(self) -> List[int]:
        return self.sample_idxs

    def get_image_path(self, idx):
        info = self.info_list[idx]
        # compute image path
        offset = os.path.join(info[0], f"{self.framedata_color_name[info[3]]}_{info[2]}.png")
        image_path = os.path.join(self.root, "stream_release_v2", offset)
        return image_path

    def get_image(self, idx):
        path = self.get_image_path(idx)
        image = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return image

    def get_rawimage_size(self, idx):
        # MUST (W, H)
        return (848, 480)

    def get_image_mask(self, idx):
        # mask_path = os.path.join(self.root, "mask", f"{self.info_str_list[idx]}.png")
        # mask = np.array(imageio.imread(mask_path, as_gray=True), dtype=np.uint8)
        # return mask
        return np.zeros((480, 848), dtype=np.uint8)

    def get_cam_intr(self, idx):
        cam_path = os.path.join(self.root, "anno", "cam_intr", f"{self.info_str_list[idx]}.pkl")
        with open(cam_path, "rb") as f:
            cam_intr = pickle.load(f)
        return cam_intr

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_hand_faces(self, idx):
        return self.rMANO.get_mano_closed_faces().numpy()

    def get_joints_3d(self, idx):
        joints_path = os.path.join(self.root, "anno", "hand_j", f"{self.info_str_list[idx]}.pkl")
        with open(joints_path, "rb") as f:
            joints_3d = pickle.load(f)
        return joints_3d

    def get_verts_3d(self, idx):
        verts_path = os.path.join(self.root, "anno", "hand_v", f"{self.info_str_list[idx]}.pkl")
        with open(verts_path, "rb") as f:
            verts_3d = pickle.load(f)
        return verts_3d

    def get_joints_2d(self, idx):
        cam_intr = self.get_cam_intr(idx)
        joints_3d = self.get_joints_3d(idx)
        return persp_project(joints_3d, cam_intr)

    def get_joints_uvd(self, idx):
        uv = self.get_joints_2d(idx)
        d = self.get_joints_3d(idx)[:, 2:]  # (21, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_uvd(self, idx):
        v3d = self.get_verts_3d(idx)
        intr = self.get_cam_intr(idx)
        uv = persp_project(v3d, intr)[:, :2]
        d = v3d[:, 2:]  # (778, 1)
        uvd = np.concatenate((uv, d), axis=1)
        return uvd

    def get_verts_2d(self, idx):
        cam_intr = self.get_cam_intr(idx)
        verts_3d = self.get_verts_3d(idx)
        return persp_project(verts_3d, cam_intr)

    def get_sides(self, idx):
        return "right"

    def get_bone_scale(self, idx):
        raise NotImplementedError(f"{self.name} does not support bone scale")

    def get_bbox_center_scale(self, idx):
        joints_2d = self.get_joints_2d(idx)
        center = get_annot_center(joints_2d)
        scale = get_annot_scale(joints_2d)
        return center, scale

    def get_mano_pose(self, idx):
        general_info_path = os.path.join(self.root, "anno", "general_info", f"{self.info_str_list[idx]}.pkl")
        with open(general_info_path, "rb") as f:
            general_info = pickle.load(f)
        raw_hand_anno = general_info["hand_anno"]

        raw_hand_pose = (raw_hand_anno["hand_pose"]).reshape((16, 4))  # quat (16, 4)
        _wrist, _remain = raw_hand_pose[0, :], raw_hand_pose[1:, :]
        cam_extr = general_info["cam_extr"]  # SE3 (4, 4))
        extr_R = cam_extr[:3, :3]  # (3, 3)

        wrist_R = extr_R.matmul(quat_to_rotmat(_wrist))  # (3, 3)
        wrist = rotmat_to_aa(wrist_R).unsqueeze(0).numpy()  # (1, 3)
        remain = quat_to_aa(_remain).numpy()  # (15, 3)
        hand_pose = np.concatenate([wrist, remain], axis=0)  # (16, 3)

        return hand_pose.astype(np.float32)

    def get_mano_shape(self, idx):
        general_info_path = os.path.join(self.root, "anno", "general_info", f"{self.info_str_list[idx]}.pkl")
        with open(general_info_path, "rb") as f:
            general_info = pickle.load(f)
        raw_hand_anno = general_info["hand_anno"]
        hand_shape = raw_hand_anno["hand_shape"].numpy().astype(np.float32)
        return hand_shape

    # for mv dataset
    def get_cam_extr(self, idx):
        general_info_path = os.path.join(self.root, "anno", "general_info", f"{self.info_str_list[idx]}.pkl")
        with open(general_info_path, "rb") as f:
            general_info = pickle.load(f)
        cam_extr = general_info["cam_extr"].numpy().astype(np.float32)
        return cam_extr

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.info_str_list[idx]}"
        return res

    def _getitem_3d(self, idx):
        # Support FreiHAND, HO3D, DexYCB
        idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)
        bbox_center, bbox_scale = self.get_bbox_center_scale(idx)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        cam_intr = self.get_cam_intr(idx)
        cam_center = self.get_cam_center(idx)
        joints_3d = self.get_joints_3d(idx)
        verts_3d = self.get_verts_3d(idx)
        joints_2d = self.get_joints_2d(idx)
        verts_uvd = self.get_verts_uvd(idx)
        joints_uvd = self.get_joints_uvd(idx)

        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)

        image_mask = self.get_image_mask(idx)

        # for mv dataset
        cam_extr = self.get_cam_extr(idx)

        flip_hand = True if hand_side != self.sides else False

        # Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_3d = self.flip_3d(joints_3d)
            verts_3d = self.flip_3d(verts_3d)
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            joints_2d = self.flip_2d(joints_2d, centerX=raw_size[0])
            image = image[:, ::-1, :]
            image_mask = image_mask[:, ::-1]

        label = {
            "idx": idx,
            "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_intr": cam_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "joints_uvd": joints_uvd,
            "verts_uvd": verts_uvd,
            "image_path": image_path,
            "raw_size": raw_size,
            "image_mask": image_mask,
            "cam_extr": cam_extr,
        }

        results = self.transform(image, label)
        results.update(label)
        return results

    def _getitem_3d_pack(self, idx):
        idx = self.get_sample_idx()[idx]
        hand_side = self.get_sides(idx)
        # load pack
        pack_path = os.path.join(self.root, self.packed_key, self.split_mode, self.data_split,
                                 f"{self.info_str_list[idx]}.pkl")
        with open(pack_path, "rb") as f:
            packed = pickle.load(f)

        cam_intr = np.array(packed["cam_intr"])
        joints_3d = np.array(packed["hand_j"])
        verts_3d = np.array(packed["hand_v"])
        joints_2d = persp_project(joints_3d, cam_intr)
        verts_2d = persp_project(verts_3d, cam_intr)
        joints_uvd = np.concatenate((joints_2d, joints_3d[:, 2:]), axis=1)
        verts_uvd = np.concatenate((verts_2d, verts_3d[:, 2:]), axis=1)

        bbox_center = get_annot_center(joints_2d)
        bbox_scale = get_annot_scale(joints_2d)
        bbox_scale = bbox_scale * self.bbox_expand_ratio  # extend bbox sacle
        cam_center = np.array([cam_intr[0, 2], cam_intr[1, 2]])

        image_path = self.get_image_path(idx)
        image = self.get_image(idx)
        raw_size = [image.shape[1], image.shape[0]]  # (W, H)
        joints_vis = self.get_joints_2d_vis(joints_2d=joints_2d, raw_size=raw_size)

        image_mask = self.get_image_mask(idx)

        flip_hand = True if hand_side != self.sides else False

        # for mv dataset
        cam_extr = np.array(packed["cam_extr"])

        # Flip 2d if needed
        if flip_hand:
            bbox_center[0] = raw_size[0] - bbox_center[0]  # image center
            joints_3d = self.flip_3d(joints_3d)
            verts_3d = self.flip_3d(verts_3d)
            joints_uvd = self.flip_2d(joints_uvd, raw_size[0])
            verts_uvd = self.flip_2d(verts_uvd, raw_size[0])
            joints_2d = self.flip_2d(joints_2d, centerX=raw_size[0])
            image = image[:, ::-1, :]
            image_mask = image_mask[:, ::-1]

        label = {
            "idx": idx,
            "cam_center": cam_center,
            "bbox_center": bbox_center,
            "bbox_scale": bbox_scale,
            "cam_intr": cam_intr,
            "joints_2d": joints_2d,
            "joints_3d": joints_3d,
            "verts_3d": verts_3d,
            "joints_vis": joints_vis,
            "joints_uvd": joints_uvd,
            "verts_uvd": verts_uvd,
            "image_path": image_path,
            "raw_size": raw_size,
            "image_mask": image_mask,
            "cam_extr": cam_extr,
        }

        results = self.transform(image, label)
        results.update(label)
        return results


@DATASET.register_module()
class OakInkMultiView(torch.utils.data.Dataset):

    @staticmethod
    def index_info_list(oiset):
        info_list = oiset.info_list
        index_map = {}
        for idx, info_item in enumerate(info_list):
            pk, sid, fid, vid = info_item
            sample_tuple = (pk, sid, fid)
            if sample_tuple not in index_map:
                index_map[sample_tuple] = [-1, -1, -1, -1]
            index_map[sample_tuple][vid] = idx
        # sanity check
        for k, v in index_map.items():
            assert v[0] != -1
            assert v[1] != -1
            assert v[2] != -1
            assert v[3] != -1
        return index_map

    def __init__(self, cfg) -> None:
        super().__init__()

        self.name = self.__class__.__name__

        # data settings
        # these code present here for narrowing check
        # as multiview dataset propose a more strict constraint over config
        self.data_mode = cfg.DATA_MODE
        self.use_quarter = cfg.get("USE_QUARTER", False)
        self.skip_frames = cfg.get("SKIP_FRAMES", 0)
        if self.use_quarter is True:
            warnings.warn(f"use_quarter is deprecated, use skip_frames=3 instead")
            self.skip_frames = 3

        assert self.data_mode == "3D", f"{self.name} only support 3D data mode"
        self.data_split = cfg.DATA_SPLIT
        assert self.data_split in [
            "all",
            "train+val",
            "train",
            "val",
            "test",
        ], "OakInk data_split must be one of <see the src>"
        self.split_mode = cfg.SPLIT_MODE
        assert self.split_mode in [
            "subject",
            "object",
        ], "OakInk split_mode must be one of ['subject', 'object]"

        # multiview settings
        self.n_views = cfg.N_VIEWS
        assert self.n_views == 4, f"{self.name} only support 4 view"

        self.test_with_multiview = cfg.get("TEST_WITH_MULTIVIEW", False)
        self.master_system = cfg.get("MASTER_SYSTEM", "as_constant_camera")
        assert self.master_system in [
            "as_constant_camera",
        ], f"{self.name} got unsupport master system mode {self.master_system}"

        self.const_cam_view_id = 0

        # preset
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX

        # load dataset (single view for given split)
        self._dataset = OakInk(cfg)

        # filter info_list
        self._index_map = self.index_info_list(self._dataset)
        self._mv_list = list(self._index_map.values())
        # self.len = len(self._mv_list)

        if self.skip_frames != 0:
            self.valid_sample_idx_list = [i for i in range(len(self._mv_list)) if i % (self.skip_frames + 1) == 0]
        else:
            self.valid_sample_idx_list = [i for i in range(len(self._mv_list))]

        self.len = len(self.valid_sample_idx_list)
        logger.warning(f"{self.name} {self.split_mode}_{self.data_split} Init Done. "
                       f"Skip frames: {self.skip_frames}. Total {self.len} samples")

    def __len__(self):
        return self.len

    def __getitem__(self, sample_idx):
        sample_idx = self.valid_sample_idx_list[sample_idx]
        idx_list = self._mv_list[sample_idx]

        # load all samples
        sample = {}
        sample["sample_idx"] = idx_list
        sample["target_cam_extr"] = []

        for internal_idx in idx_list:
            internal_label = self._dataset[internal_idx]
            sample["target_cam_extr"].append(internal_label["cam_extr"])

            for query in [
                    "rot_rad",
                    "rot_mat3d",
                    "affine",
                    "image",
                    "target_bbox_center",
                    'target_bbox_scale',
                    'target_joints_2d',
                    'target_joints_vis',
                    'target_root_d',
                    'target_joints_uvd',
                    'target_verts_uvd',
                    'affine_postrot',
                    'target_cam_intr',
                    'target_joints_3d',
                    'target_verts_3d',
                    'target_joints_3d_rel',
                    'target_verts_3d_rel',
                    'target_root_joint',
                    "target_joints_3d_no_rot",
                    "target_verts_3d_no_rot",
                    'idx',
                    'cam_center',
                    'bbox_center',
                    'bbox_scale',
                    'cam_intr',
                    'joints_2d',
                    'joints_3d',
                    'verts_3d',
                    'joints_vis',
                    'joints_uvd',
                    'verts_uvd',
                    'image_path',
                    'raw_size',
                    'image_mask',
                    "extr_prerot",
                    'target_joints_heatmap',
            ]:
                if query not in sample:
                    sample[query] = [internal_label[query]]
                else:
                    sample[query].append(internal_label[query])

        if self.master_system == "as_constant_camera":
            master_id = 0
            T_master_2_new_master = sample["target_cam_extr"][master_id].copy()
            master_joints_3d = sample["target_joints_3d_no_rot"][master_id]
            master_verts_3d = sample["target_verts_3d_no_rot"][master_id]
        else:
            pass  # should not go here

        for i, T_m2c in enumerate(sample["target_cam_extr"]):
            # we request inverse to be extr here (tf from camspace to master instead of tf from master to camspace)
            # i.e. (Extr @ p == p_master)
            # T_new_master_2_cam = np.linalg.inv(T_m2c @ np.linalg.inv(T_master_2_new_master))
            extr_prerot = sample["extr_prerot"][i]
            extr_prerot_tf_inv = np.eye(4).astype(extr_prerot.dtype)
            extr_prerot_tf_inv[:3, :3] = extr_prerot.T
            T_new_master_2_cam = T_master_2_new_master @ np.linalg.inv(T_m2c)
            # rotate the point then transform to new master
            # note we request the inverse to be extr here (tf from camspace 2 master)
            # (Extr @ R_aug_inv @ p == p_master)
            sample["target_cam_extr"][i] = T_new_master_2_cam @ extr_prerot_tf_inv

        for query in sample.keys():
            if isinstance(sample[query][0], (int, float, np.ndarray, torch.Tensor)):
                sample[query] = np.stack(sample[query])

        sample["master_id"] = master_id
        sample["master_joints_3d"] = master_joints_3d
        sample["master_verts_3d"] = master_verts_3d

        return sample


@DATASET.register_module()
class OakInkMultiView_Video(OakInkMultiView):

    def __init__(self, cfg):
        cfg.USE_QUARTER = False
        cfg.SKIP_FRAMES = 0
        super().__init__(cfg)

        self.name = type(self).__name__
        # self.cfg = cfg
        self.seq_len = cfg.SEQ_LEN
        self.drop_last_frames = cfg.get("DROP_LAST_FRAMES", True)
        self.interval_frames = cfg.get("INTERVAL_FRAMES", 0)

        self._load_video_frames()

        assert self.master_system == "as_constant_camera", f"{self.name} only support master system mode 'as_constant_camera' "
        logger.warning(
            f"{self.name} {self.split_mode}_{self.data_split} Init Done. {len(self.multiview_video_sample_idxs)} samples"
        )

    def _load_video_frames(self):
        self.multiview_video_sample_idxs = []

        if self.split_mode == 'object':
            if self.data_split == 'train+val':
                with open('./assets/video_task/oakink_multiview_video_idxs_train+val_object.pkl', 'rb') as f_idx:
                    all_multiview_samples = pickle.load(f_idx)
            elif self.data_split == 'test':
                with open('./assets/video_task/oakink_multiview_video_idxs_test_object.pkl', 'rb') as f_idx:
                    all_multiview_samples = pickle.load(f_idx)
            else:
                raise ValueError(f"Don't supported data_split: {self.data_split}")
        else:
            raise ValueError(f"Don't supported split_mode: {self.split_mode}")

        if self.interval_frames != 0:
            all_multiview_samples = all_multiview_samples[::self.interval_frames]

        self.multiview_video_sample_idxs = []
        for i in range(len(all_multiview_samples)):
            if all_multiview_samples[i][-1][:2] == all_multiview_samples[i + self.seq_len - 1][-1][:2]:
                tmp_list = []
                for j in range(i, i + self.seq_len):
                    tmp_list.append(all_multiview_samples[j])
                self.multiview_video_sample_idxs.append(tmp_list)
            if i + self.seq_len == len(all_multiview_samples):
                break

    def __len__(self):
        return len(self.multiview_video_sample_idxs)

    def _save_tmp(self, idx):
        idx_list = self._mv_list[idx]
        mv_list_k = list(self._index_map.keys())
        mv_list_v = list(self._index_map.values())

        seq_names = [mv_list_v[idx][i] for i in range(len(mv_list_v[idx]))]
        single_idx = [mv_list_k[idx][0], mv_list_k[idx][1], mv_list_k[idx][2]]
        save_list = [idx, seq_names, single_idx]  # [[idx,[single_idxs],[pk, sid, fid]] * seq_len]
        return save_list

    def __getitem__(self, idx):
        multiview_video_id_list = self.multiview_video_sample_idxs[idx]  # [[idx,[single_idxs],[seq_names]] * seq_len]
        sample = dict()
        for i in range(self.seq_len):
            # get sample from the source set. (OakInkMultiView's getitem)
            multiview_idx = multiview_video_id_list[i][0]
            i_sample = super().__getitem__(multiview_idx)
            for query, value in i_sample.items():
                if query in sample:
                    sample[query].append(value)
                else:
                    sample[query] = [value]

        return sample
