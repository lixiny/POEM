import hashlib
import json
import os
import pickle
import random
import warnings
from typing import List

import copy
import imageio
import numpy as np
import torch
import torch.nn as nn
import yaml
from manotorch.manolayer import ManoLayer, MANOOutput
from scipy.spatial.distance import cdist
from termcolor import colored

from ..utils.builder import DATASET
from ..utils.config import CN
from ..utils.etqdm import etqdm
from ..utils.logger import logger
from ..utils.transform import (batch_ref_bone_len, bbox_xywh_to_xyxy, cal_transform_mean, get_annot_center,
                               get_annot_scale, persp_project, SE3_transform)
from .hdata import HDataset, kpId2vertices


@DATASET.register_module()
class DexYCB(HDataset):

    def __init__(self, cfg):
        super().__init__(cfg)

        if 'SPLIT_MODE' in cfg:
            warnings.warn("SPLIT_MODE is deprecated, use `SETUP` instead")
            if 'SETUP' in cfg:
                assert cfg.SETUP == cfg.SPLIT_MODE, "SETUP and SPLIT_MODE must be the same"
                self.setup = cfg.SETUP
            else:
                self.setup = cfg.SPLIT_MODE
        else:
            self.setup = cfg.SETUP

        self.use_left_hand = cfg.USE_LEFT_HAND
        self.filter_invisible_hand = cfg.FILTER_INVISIBLE_HAND
        self.dex_ycb = None

        self.dexycb_mano_right = ManoLayer(
            flat_hand_mean=False,
            side="right",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        )
        self.dexycb_mano_left = ManoLayer(
            flat_hand_mean=False,
            side="left",
            mano_assets_root="assets/mano_v1_2",
            use_pca=True,
            ncomps=45,
        ) if self.use_left_hand else None

        self.load_dataset()

    def _preload(self):
        self.name = "DexYCB"
        self.root = os.path.join(self.data_root, self.name)
        os.environ["DEX_YCB_DIR"] = self.root

        self.cache_identifier_dict = {
            "data_split": self.data_split,
            "setup": self.setup,
            "use_left_hand": self.use_left_hand,
            "filter_invisible_hand": self.filter_invisible_hand,
        }
        self.cache_identifier_raw = json.dumps(self.cache_identifier_dict, sort_keys=True)
        self.cache_identifier = hashlib.md5(self.cache_identifier_raw.encode("ascii")).hexdigest()
        self.cache_path = os.path.join("common", "cache", self.name, "{}.pkl".format(self.cache_identifier))

    def load_dataset(self):
        from dex_ycb_toolkit.dex_ycb import DexYCBDataset
        from dex_ycb_toolkit.factory import get_dataset

        self._preload()
        cache_folder = os.path.dirname(self.cache_path)
        os.makedirs(cache_folder, exist_ok=True)

        dexycb_name = f"{self.setup}_{self.data_split}"
        logger.info(f"DexYCB use split: {dexycb_name}")
        self.dex_ycb: DexYCBDataset = get_dataset(dexycb_name)
        self.raw_size = (640, 480)

        # region filter sample
        if self.use_cache and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as p_f:
                self.sample_idxs = pickle.load(p_f)
            logger.info(f"Loaded cache for {self.name}_{self.data_split}_{self.setup} from {self.cache_path}")
        else:
            self.sample_idxs = []
            for i, (seq_id, cam_id, frame_id) in enumerate(etqdm(self.dex_ycb._mapping)):
                if self.dex_ycb._mano_side[seq_id] == "left" and not self.use_left_hand:
                    continue

                label_file = os.path.join(self.dex_ycb._data_dir, self.dex_ycb._sequences[seq_id],
                                          self.dex_ycb._serials[cam_id], self.dex_ycb._label_format.format(frame_id))
                joint_2d = np.load(label_file)["joint_2d"]
                if np.any(joint_2d == -1):
                    continue

                self.sample_idxs.append(i)

            if self.use_cache:
                with open(self.cache_path, "wb") as p_f:
                    pickle.dump(self.sample_idxs, p_f)
                logger.info(f"Wrote cache for {self.name}_{self.data_split}_{self.setup} to {self.cache_path}")
        # endregion
        logger.info(f"{self.name} Got {colored(len(self.sample_idxs), 'yellow', attrs=['bold'])}"
                    f"/{len(self.dex_ycb)} samples for data_split {self.data_split}")

    def __len__(self):
        return len(self.sample_idxs)

    def get_sample_idx(self) -> List[int]:
        return self.sample_idxs

    # @lru_cache(maxsize=None)
    def get_label(self, label_file: str):
        return np.load(label_file)

    def get_image(self, idx):
        path = self.get_image_path(idx)
        img = np.array(imageio.imread(path, pilmode="RGB"), dtype=np.uint8)
        return img

    def get_image_path(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        return sample["color_file"]

    def get_rawimage_size(self, idx):
        return (640, 480)

    def get_cam_center(self, idx):
        intr = self.get_cam_intr(idx)
        return np.array([intr[0, 2], intr[1, 2]])

    def get_image_mask(self, idx):
        image_path = self.get_image_path(idx)
        mask_path = image_path.replace("color", "mask")
        mask_path = mask_path.replace("DexYCB", "DexYCB_supp")
        img = np.array(imageio.imread(mask_path, as_gray=True), dtype=np.uint8)
        return img

    def get_cam_intr(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        return np.array(
            [
                [sample["intrinsics"]["fx"], 0.0, sample["intrinsics"]["ppx"]],
                [0.0, sample["intrinsics"]["fy"], sample["intrinsics"]["ppy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def get_bbox_center_scale(self, idx):
        joints2d = self.get_joints_2d(idx)  # (21, 2)
        center = get_annot_center(joints2d)
        scale = get_annot_scale(joints2d)
        return center, scale

    def get_hand_faces(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        assert sample["mano_side"] == "right"
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        faces = np.array(mano_layer.get_mano_closed_faces()).astype(np.long)
        return faces

    def get_verts_3d(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"])
        shape = torch.tensor(sample["mano_betas"]).unsqueeze(0)
        mano_layer = self.dexycb_mano_left if sample["mano_side"] == "left" else self.dexycb_mano_right
        mano_out: MANOOutput = mano_layer(pose_m[:, :48], shape)
        hand_verts = mano_out.verts + pose_m[:, 48:]
        return hand_verts.squeeze(0).numpy().astype(np.float32)

    def get_verts_2d(self, idx):
        verts_3d = self.get_verts_3d(idx)
        cam_intr = self.get_cam_intr(idx)
        return persp_project(verts_3d, cam_intr)

    def get_joints_2d(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_2d"].squeeze(0)

    def get_joints_3d(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        return label["joint_3d"].squeeze(0)

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

    def get_bone_scale(self, idx):
        joints_3d = self.get_joints_3d(idx)
        bone_len = batch_ref_bone_len(np.expand_dims(joints_3d, axis=0)).squeeze(0)
        return bone_len.astype(np.float32)

    def get_sample_identifier(self, idx):
        res = f"{self.name}__{self.cache_identifier_raw}__{idx}"
        return res

    def get_sides(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        return sample["mano_side"]

    def get_mano_pose(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        label = self.get_label(sample["label_file"])  # keys: seg, pose_y, pose_m, joint_3d, joint_2d
        pose_m = torch.from_numpy(label["pose_m"])
        mano_layer = self.dexycb_mano_right if sample["mano_side"] == "right" else self.dexycb_mano_left
        pose = mano_layer.rotation_by_axisang(pose_m[:, :48])["full_poses"]  # (1, 48)
        pose = pose.squeeze(0).numpy().astype(np.float32)
        return pose

    def get_mano_shape(self, idx):
        ori_idx = self.sample_idxs[idx]
        sample = self.dex_ycb[ori_idx]
        shape = sample["mano_betas"]
        shape = np.array(shape, dtype=np.float32)
        return shape


@DATASET.register_module()
class DexYCBMultiView(torch.utils.data.Dataset):

    def __init__(self, cfg):
        super().__init__()

        self.name = type(self).__name__
        self.cfg = cfg
        self.n_views = cfg.N_VIEWS
        self.data_split = cfg.DATA_SPLIT
        self.skip_frames = cfg.get("SKIP_FRAMES", 0)
        assert self.data_split in ["train", "val", "test"], f"{self.name} unsupport data split {self.data_split}"
        self.test_with_multiview = cfg.get("TEST_WITH_MULTIVIEW", False)

        self.master_system = cfg.get("MASTER_SYSTEM", "default")
        if self.master_system == "default":
            self.master_system = "as_first_camera"
            warnings.warn(f"MultiView dataset require you to specify the filed: MASTER_SYSTEM in config file."
                          f"Currently the default value: as_first_camera is used, but may not match your intention.")

        assert self.master_system in [
            "as_first_camera",
            "as_constant_camera",
        ], f"{self.name} got unsupport master system mode {self.master_system}"
        logger.warning(f"{self.name} use master system {self.master_system}")

        self.CONST_CAM_SERIAL = "840412060917"

        self.data_mode = cfg.DATA_MODE
        assert self.data_mode == "3D", f"{self.name} only support 3D data mode"
        self.setup = cfg.SETUP
        self.center_idx = cfg.DATA_PRESET.CENTER_IDX
        _trainset, _valset, _testset = self._single_view_dexycb()

        self.set_mappings = {
            f"{cfg.SETUP}_train": _trainset,
            f"{cfg.SETUP}_val": _valset,
            f"{cfg.SETUP}_test": _testset
        }
        self.root = _trainset.root

        self.multiview_sample_idxs = []
        self.multiview_sample_infos = []

        if self.setup in ['s0', 's1', 's3']:  # full view mode
            source_set_name = f"{self.setup}_{self.data_split}"  # eg s0_train
            source_set: DexYCB = self.set_mappings[source_set_name]
            multivew_mapping = {}
            for i, ori_idx in enumerate(source_set.sample_idxs):
                seq_id, cam_id, frame_id = source_set.dex_ycb._mapping[ori_idx]
                if (seq_id, frame_id) not in multivew_mapping:
                    multivew_mapping[(seq_id, frame_id)] = [(cam_id, i)]
                else:
                    multivew_mapping[(seq_id, frame_id)].append((cam_id, i))

            for key, value in multivew_mapping.items():
                seq_id, frame_id = key
                self.multiview_sample_idxs.append([i for (_, i) in value])
                self.multiview_sample_infos.append([{
                    "set_name": source_set_name,
                    "seq_id": seq_id,
                    "seq_name": source_set.dex_ycb._sequences[seq_id],
                    "cam_id": cam_id,
                    "cam_serial": source_set.dex_ycb._serials[cam_id],
                    "frame_id": frame_id,
                } for (cam_id, _) in value])
        elif self.setup == 's2':  # unseen view mode
            # TODO: ablation
            raise NotImplementedError("s2 is not implemented")
        else:
            raise ValueError(f"{self.setup} is not supported")

        if self.data_split == "test" and not self.test_with_multiview:
            warnings.warn(f"test_with_multiview = {self.test_with_multiview} is dreprecated and will be removed soon")
            num_multiview_samples = len(self.multiview_sample_idxs)
            for _ in range(num_multiview_samples):
                curr_sample_idxs = self.multiview_sample_idxs.pop(0)  # a list of 8 elements: sample idx
                curr_sample_infos = self.multiview_sample_infos.pop(0)

                self.multiview_sample_idxs.append(copy.deepcopy(curr_sample_idxs))
                self.multiview_sample_infos.append(copy.deepcopy(curr_sample_infos))

                num_views_per_sample = len(curr_sample_idxs)
                for j in range(num_views_per_sample - 1):
                    sample_idx = curr_sample_idxs.pop(0)
                    sample_info = curr_sample_infos.pop(0)
                    curr_sample_idxs.append(sample_idx)
                    curr_sample_infos.append(sample_info)
                    self.multiview_sample_idxs.append(copy.deepcopy(curr_sample_idxs))
                    self.multiview_sample_infos.append(copy.deepcopy(curr_sample_infos))

        total_len = len(self.multiview_sample_idxs)
        if self.skip_frames != 0:
            self.valid_sample_idx_list = [i for i in range(total_len) if i % (self.skip_frames + 1) == 0]
        else:
            self.valid_sample_idx_list = [i for i in range(total_len)]

        self.len = len(self.valid_sample_idx_list)
        logger.warning(f"{self.name} {self.setup}_{self.data_split} Init Done. "
                       f"Skip frames: {self.skip_frames}, total {self.len} samples")

    def _single_view_dexycb(self):
        cfg_train = dict(
            TYPE="DexYCB",
            DATA_SPLIT="train",
            DATA_MODE=self.data_mode,
            SETUP=self.setup,
            DATA_ROOT=self.cfg.DATA_ROOT,
            USE_LEFT_HAND=self.cfg.USE_LEFT_HAND,
            FILTER_INVISIBLE_HAND=self.cfg.FILTER_INVISIBLE_HAND,
            TRANSFORM=self.cfg.TRANSFORM,
            DATA_PRESET=self.cfg.DATA_PRESET,
        )

        cfg_val, cfg_test = cfg_train.copy(), cfg_train.copy()
        cfg_val["DATA_SPLIT"] = "val"
        cfg_test["DATA_SPLIT"] = "test"

        dex_ycb_train = DexYCB(CN(cfg_train))
        dex_ycb_val = DexYCB(CN(cfg_val))
        dex_ycb_test = DexYCB(CN(cfg_test))

        return dex_ycb_train, dex_ycb_val, dex_ycb_test

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        idx = self.valid_sample_idx_list[idx]
        multiview_id_list = self.multiview_sample_idxs[idx]
        multiview_info_list = self.multiview_sample_infos[idx]

        if self.master_system == "as_first_camera":
            # NOTE: shuffle the order of cameras to make the ``first camera`` in training mode not always the same
            if self.data_split == "train":
                ## NOTE: shulffling the order the two list below together !!!
                lists_to_shuffle = list(zip(multiview_id_list, multiview_info_list))
                random.shuffle(lists_to_shuffle)
                multiview_id_list, multiview_info_list = zip(*lists_to_shuffle)
        elif self.master_system == "as_constant_camera":
            # NOTE: acquire the const camera id and info and put them to the first position
            const_v_id = -101
            for vi, info in enumerate(multiview_info_list):
                if info["cam_serial"] == self.CONST_CAM_SERIAL:
                    const_v_id = vi
                    break
            assert const_v_id != -101, f"Cannot find the constant camera serial {self.CONST_CAM_SERIAL} in the dataset"
            curr_idx = multiview_id_list.pop(const_v_id)
            curr_info = multiview_info_list.pop(const_v_id)
            multiview_id_list.insert(0, curr_idx)
            multiview_info_list.insert(0, curr_info)

        src_seq_dir = os.path.join(self.root, multiview_info_list[0]["seq_name"])
        meta_path = os.path.join(src_seq_dir, "meta.yml")
        with open(meta_path) as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        extr_file_id = meta["extrinsics"]
        extr_file_path = os.path.join(self.root, "calibration", f"extrinsics_{extr_file_id}", "extrinsics.yml")
        with open(extr_file_path) as f:
            extr_dict = yaml.load(f, Loader=yaml.FullLoader)

        # master_serial = extr_dict["master"]
        serial_extr_mapping = {}
        for serial, raw in extr_dict["extrinsics"].items():
            T = np.asarray(raw, dtype=np.float32).reshape(3, 4)
            T = np.concatenate([T, np.array([[0, 0, 0, 1]])], axis=0).astype(np.float32)
            serial_extr_mapping[serial] = T

        sample = dict()
        sample["sample_idx"] = multiview_id_list
        sample["target_cam_extr"] = list()
        sample["cam_serial"] = list()

        for i, info in zip(multiview_id_list, multiview_info_list):
            # get the source set -- one of the  DexYCB s#_train, s#_val and s#_test.
            source_set = self.set_mappings[info["set_name"]]

            cam_serial = info["cam_serial"]
            T_master_2_cam = serial_extr_mapping[cam_serial]
            sample["target_cam_extr"].append(T_master_2_cam)
            sample["cam_serial"].append(cam_serial)

            # get sample from the single viwe source set. (DexYCB's getitem)
            src_sample = source_set[i]
            for query, value in src_sample.items():
                if query in sample:
                    sample[query].append(value)
                else:
                    sample[query] = [value]
        """
        set a new master serial
        each trasnf in sample["target_cam_extr"] is a T: mater_2_cam:  
        >>>>  P_in_master = T_mater_2_cam @ P_in_cam
        """
        if self.master_system == "as_first_camera":
            new_master_id = 0
            new_master_serial = sample["cam_serial"][new_master_id]
            T_master_2_new_master = sample["target_cam_extr"][new_master_id]
            master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
            master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]

        elif self.master_system == "as_constant_camera":
            new_master_serial = self.CONST_CAM_SERIAL
            new_master_id = sample["cam_serial"].index(new_master_serial)
            assert new_master_id == 0, f"The constant camera should already be moved to the first place in the list, got {new_master_id}"
            T_master_2_new_master = sample["target_cam_extr"][new_master_id]
            master_joints_3d = sample["target_joints_3d_no_rot"][new_master_id]
            master_verts_3d = sample["target_verts_3d_no_rot"][new_master_id]

        for i, T_m2c in enumerate(sample["target_cam_extr"]):
            T_new_master_2_cam = np.linalg.inv(T_master_2_new_master) @ T_m2c
            extr_prerot = sample["extr_prerot"][i]  # (3, 3)
            extr_pre_transf = np.concatenate([extr_prerot, np.zeros((3, 1))], axis=1)
            extr_pre_transf = np.concatenate([extr_pre_transf, np.array([[0, 0, 0, 1]])], axis=0)

            T_new_master_2_cam = np.linalg.inv(extr_pre_transf @ np.linalg.inv(T_new_master_2_cam))
            sample["target_cam_extr"][i] = T_new_master_2_cam.astype(np.float32)

        for query in sample.keys():
            if isinstance(sample[query][0], (int, float, np.ndarray, torch.Tensor)):
                sample[query] = np.stack(sample[query])

        sample["master_id"] = new_master_id
        sample["master_serial"] = new_master_serial
        sample["master_joints_3d"] = master_joints_3d
        sample["master_verts_3d"] = master_verts_3d

        return sample


@DATASET.register_module()
class DexYCBMultiView_Video(DexYCBMultiView):

    def __init__(self, cfg):
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
            f"{self.name} {self.setup}_{self.data_split} Init Done. {len(self.multiview_video_sample_idxs)} samples")

    def _load_video_frames(self):
        if self.data_split == 'train':
            with open('./assets/video_task/dexycb_multiview_video_idxs_train.pkl', 'rb') as f_idx:
                all_multiview_samples = pickle.load(f_idx)
        elif self.data_split == 'test':
            with open('./assets/video_task/dexycb_multiview_video_idxs_test.pkl', 'rb') as f_idx:
                all_multiview_samples = pickle.load(f_idx)
        else:
            raise ValueError(f"Don't supported data_split: {self.data_split}")
        if self.interval_frames != 0:
            all_multiview_samples = all_multiview_samples[::self.interval_frames]

        self.multiview_video_sample_idxs = []
        for i in range(len(all_multiview_samples)):
            if all_multiview_samples[i][-1] == all_multiview_samples[i + self.seq_len - 1][-1]:
                tmp_list = []
                for j in range(i, i + self.seq_len):
                    tmp_list.append(all_multiview_samples[j])
                self.multiview_video_sample_idxs.append(tmp_list)
            if i + self.seq_len == len(all_multiview_samples):
                break

    def __len__(self):
        return len(self.multiview_video_sample_idxs)

    def _save_tmp(self, idx):
        multiview_id_list = self.multiview_sample_idxs[idx]
        multiview_info_list = self.multiview_sample_infos[idx]
        tmp = []
        for i in multiview_info_list:
            tmp.append(i['seq_name'])
        save_list = []
        save_list.append(idx)
        save_list.append(multiview_id_list)
        save_list.append(tmp)
        # print(save_list)
        return save_list

    def __getitem__(self, idx):
        multiview_video_id_list = self.multiview_video_sample_idxs[idx]  # [[idx,[single_idxs],[seq_names]] * seq_len]
        sample = dict()
        for i in range(self.seq_len):
            # get sample from the source set. (HO3Dv3MultiView's getitem)
            multiview_idx = multiview_video_id_list[i][0]
            i_sample = super().__getitem__(multiview_idx)
            for query, value in i_sample.items():
                if query in sample:
                    sample[query].append(value)
                else:
                    sample[query] = [value]
        return sample
