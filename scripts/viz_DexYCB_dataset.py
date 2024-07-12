print("hello world")
import time

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as tvF
from lib.datasets.dexycb import DexYCB
from lib.models.layers.mano_wrapper import MANO
from lib.utils.etqdm import etqdm
from lib.utils.heatmap import sample_with_heatmap
from lib.utils.neural_renderer import NeuralRenderer
from lib.utils.transform import bchw_2_bhwc, denormalize, persp_project
from lib.viztools.draw import plot_hand
from lib.utils.config import CN

DEXYCB_UVD_CONFIG = dict(
    DATA_MODE="UVD",
    DATA_ROOT="data",
    SETUP="s0",
    USE_LEFT_HAND=False,
    FILTER_INVISIBLE_HAND=True,
    TRANSFORM=dict(
        TYPE="SimpleTransformUVD",
        AUG=True,
        SCALE_JIT=0.125,
        COLOR_JIT=0.3,
        ROT_JIT=30,
        ROT_PROB=1.0,
        OCCLUSION=False,
        OCCLUSION_PROB=0.5,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=1.7,
        IMAGE_SIZE=(224, 224),
        CENTER_IDX=0,
    ),
)

DEXYCB_3D_CONFIG = dict(
    DATA_MODE="3D",
    DATA_ROOT="data",
    SETUP="s0",
    USE_LEFT_HAND=False,
    FILTER_INVISIBLE_HAND=True,
    TRANSFORM=dict(
        TYPE="SimpleTransform3DMANO",
        AUG=True,
        SCALE_JIT=0.125,
        COLOR_JIT=0.3,
        ROT_JIT=30,
        ROT_PROB=1.0,
        OCCLUSION=True,
        OCCLUSION_PROB=0.5,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=1.7,
        IMAGE_SIZE=(224, 224),
        CENTER_IDX=0,
    ),
)

DEXYCB_2D_CONFIG = dict(
    DATA_MODE="2D",
    DATA_ROOT="data",
    SETUP="s0",
    USE_LEFT_HAND=False,
    FILTER_INVISIBLE_HAND=True,
    TRANSFORM=dict(
        TYPE="SimpleTransform2D",
        AUG=True,
        SCALE_JIT=0.125,
        COLOR_JIT=0.3,
        ROT_JIT=30,
        ROT_PROB=1.0,
        OCCLUSION=True,
        OCCLUSION_PROB=0.3,
    ),
    MINI_FACTOR=1.0,
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=1.5,
        IMAGE_SIZE=(224, 224),
        CENTER_IDX=0,
        WITH_MASK=True,
        WITH_HEATMAP=True,
        MASK_SCALE_TO_HEATMAP=True,
        HEATMAP_SIZE=(64, 64),
        HEATMAP_SIGMA=2,
    ),
)

mano_wrapper = MANO(side="right",
                    joint_rot_mode="axisang",
                    use_pca=False,
                    mano_assets_root="assets/mano_v1_2",
                    center_idx=9,
                    flat_hand_mean=True)


def viz_hdata_dexycb_3d(args):
    config = CN(DEXYCB_3D_CONFIG)
    config.DATA_SPLIT = args.split
    dexycb = DexYCB(config)

    renderer = NeuralRenderer().cuda()

    bar = etqdm(range(len(dexycb)))
    for i, _ in enumerate(bar):
        i = np.random.randint(len(dexycb))
        output = dexycb[i]
        image = denormalize(output["image"], [0.5, 0.5, 0.5], [1, 1, 1]).numpy().transpose(1, 2, 0)
        image = (image * 255.0).astype(np.uint8)
        # mask = (output["mask"] * 255.0).astype(np.uint8)

        joints_2d = output["target_joints_2d"]
        joints_uvd = output["target_joints_uvd"]
        verts_3d = output["target_verts_3d"]
        joints_3d = output["target_joints_3d"]
        cam_intr = output["target_cam_intr"]
        mano_pose = output["target_mano_pose"].reshape(-1)
        mano_shape = output["target_mano_shape"]

        mano_output = mano_wrapper.forward(
            torch.from_numpy(mano_pose[None, ...]), 
            torch.from_numpy(mano_shape[None, ...]),
        )
        verts_rel = mano_output["_verts_rel"].squeeze(0).numpy()
        center_idx = mano_wrapper.mano_layer.center_idx
        joint_root = joints_3d[center_idx, :] # (3,)
        verts_3d_from_mano = verts_rel + joint_root
        assert np.allclose(verts_3d, verts_3d_from_mano, atol=1e-5)

        joints_2d_proj = persp_project(verts_3d, cam_intr)

        # keypoints
        frame_1 = image.copy()
        # mask = mask[:, :, None]
        # mask = np.concatenate([mask, mask, mask], axis=2)
        # frame_1 = cv2.addWeighted(frame_1, 0.5, mask, 0.5, 0)
        all_2d_opt = {"gt": joints_2d, "uv": joints_2d_proj}
        for i in range(joints_2d_proj.shape[0]):
            cx = int(joints_2d_proj[i, 0])
            cy = int(joints_2d_proj[i, 1])
            cv2.circle(frame_1, (cx, cy), radius=2, thickness=-1, color=np.array([1.0, 0.0, 0.0]) * 255)
        # plot_hand(frame_1, all_2d_opt["gt"], linewidth=1)

        img_list = [image, frame_1]
        comb_image = np.hstack(img_list)
        imageio.imwrite("tmp/img.png", comb_image)
        time.sleep(2)


def viz_hdata_dexycb_uvd(args):
    config = CN(DEXYCB_UVD_CONFIG)
    config.DATA_SPLIT = args.split
    dexycb = DexYCB(config)

    renderer = NeuralRenderer().cuda()

    bar = etqdm(range(len(dexycb)))
    for i, _ in enumerate(bar):
        i = np.random.randint(len(dexycb))
        output = dexycb[i]
        image = denormalize(output["image"], [0.5, 0.5, 0.5], [1, 1, 1]).numpy().transpose(1, 2, 0)
        image = (image * 255.0).astype(np.uint8)
        # mask = (output["mask"] * 255.0).astype(np.uint8)

        joints_2d = output["target_joints_2d"]
        joints_uvd = output["target_joints_uvd"]
        verts_uvd = output["target_verts_uvd"]

        # uvd
        W, H = image.shape[1], image.shape[0]
        joints_uv = joints_uvd[:, :2] * np.array([W, H])
        verts_uv = verts_uvd[:, :2] * np.array([W, H])
        verts_d = verts_uvd[:, 2:]
        verts_uvd = np.concatenate([verts_uv, verts_d], axis=1)

        # depth
        verts_uvd = torch.from_numpy(verts_uvd).unsqueeze(0).cuda().float()
        far = verts_uvd[:, :, 2].max()
        # faces = mano_wrapper.mano_layer.get_mano_closed_faces().unsqueeze(0).cuda()
        faces = torch.from_numpy(dexycb.get_hand_faces(i)).unsqueeze(0).cuda()
        scale = torch.Tensor([1.0, 1.0]).view((1, 2)).cuda()
        trans = torch.Tensor([0.0, 0.0]).view((1, 2)).cuda()
        dep, _ = renderer(
            mode="ortho",
            vertices=verts_uvd,
            faces=faces,
            rast_size=H // 2,
            orig_size=(W, H),
            anti_aliasing=True,
            far=far,
            scale=scale,
            trans=trans,
        )
        dep = dep.unsqueeze(1)
        dep = torch.nn.functional.interpolate(dep, size=(H, W))
        dep = dep.repeat(1, 3, 1, 1)  # (B, C, H, W)
        dep = bchw_2_bhwc(dep).squeeze(0)  # (H, W, C)
        dep = dep.detach().cpu().numpy() * 255
        dep = dep.astype(np.uint8)

        # keypoints
        frame_1 = image.copy()
        # mask = mask[:, :, None]
        # mask = np.concatenate([mask, mask, mask], axis=2)
        # frame_1 = cv2.addWeighted(frame_1, 0.5, mask, 0.5, 0)
        all_2d_opt = {"gt": joints_2d, "uv": joints_uv}
        plot_hand(frame_1, all_2d_opt["uv"], linewidth=2)
        # plot_hand(frame_1, all_2d_opt["gt"], linewidth=1)

        img_list = [image, dep, frame_1]
        comb_image = np.hstack(img_list)
        imageio.imwrite("tmp/img.png", comb_image)
        time.sleep(2)


def viz_hdata_dexycb_2d(args):
    config = CN(DEXYCB_2D_CONFIG)
    config.DATA_SPLIT = args.split
    dexycb = DexYCB(config)

    for i in range(len(dexycb)):
        output = dexycb[i]
        image = denormalize(output["image"], [0.5, 0.5, 0.5], [1, 1, 1]).numpy().transpose(1, 2, 0)
        image = (image * 255.0).astype(np.uint8)
        mask = (output["mask"] * 255.0).astype(np.uint8)

        img_mask = image.copy()
        mask = mask[:, :, None]
        mask = np.concatenate([mask, mask, mask], axis=2)
        mask = cv2.resize(mask, (img_mask.shape[1], img_mask.shape[0]))
        img_mask = cv2.addWeighted(img_mask, 0.5, mask, 0.5, 0)

        joints_heatmap = output["target_joints_heatmap"]
        img_hm = sample_with_heatmap(image, joints_heatmap)
        comb_img = np.hstack([img_mask, img_hm])

        imageio.imwrite("tmp/img.png", comb_img)
        time.sleep(2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-sp", "--split", type=str, default="train", help="data split type")
    parser.add_argument("-md", "--mode", type=str, default="3d", help="data split type")

    args, _ = parser.parse_known_args()
    if args.mode == "uvd":
        viz_hdata_dexycb_uvd(args)
    elif args.mode == "2d":
        viz_hdata_dexycb_2d(args)
    elif args.mode == "3d":
        viz_hdata_dexycb_3d(args)
