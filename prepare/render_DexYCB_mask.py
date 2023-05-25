import os

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from lib.datasets.dexycb import DexYCB
from lib.utils.etqdm import etqdm
from lib.utils.neural_renderer import NeuralRenderer
from lib.utils.transform import bchw_2_bhwc
from lib.utils.config import CN

DEXYCB_UVD_CONFIG = dict(
    DATA_MODE="UVD",
    DATA_ROOT="data",
    SETUP="s0",
    USE_LEFT_HAND=False,
    FILTER_INVISIBLE_HAND=True,
    TRANSFORM=dict(
        TYPE="SimpleTransform2D",
        AUG=False,
        SCALE_JIT=0,
        COLOR_JIT=0,
        ROT_JIT=0,
        ROT_PROB=0,
        OCCLUSION=False,
        OCCLUSION_PROB=0.0,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=1.7,
        IMAGE_SIZE=(224, 224),
        CENTER_IDX=0,
    ),
)

def main(args):
    config = CN(DEXYCB_UVD_CONFIG)
    config.DATA_SPLIT = args.split
    dexycb = DexYCB(config)

    renderer = NeuralRenderer().cuda()
    scale = torch.Tensor([1.0, 1.0]).view((1, 2)).cuda()
    trans = torch.Tensor([0.0, 0.0]).view((1, 2)).cuda()

    bar = etqdm(range(len(dexycb)))
    for i, _ in enumerate(bar):
        # 我们希望拿到原图和原图上的标注，因此不能调用 __getitem__函数
        image = dexycb.get_image(i)
        image_path = dexycb.get_image_path(i)
        mask_dump_path = image_path.replace("color", 'mask')
        mask_dump_path = mask_dump_path.replace("DexYCB", 'DexYCB_supp')

        os.makedirs(os.path.dirname(mask_dump_path), exist_ok=True)

        if os.path.exists(mask_dump_path):
            # logger.warning("{} already exists".format(mask_dump_path))
            continue

        W, H = dexycb.get_rawimage_size(i)
        face = torch.from_numpy(dexycb.get_hand_faces(i)).unsqueeze(0).cuda()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        verts_uvd = dexycb.get_verts_uvd(i)

        verts = torch.from_numpy(verts_uvd).unsqueeze(0).cuda()
        verts[:, :, 2] = verts[:, :, 2] + 0.5  # to meters
        far = verts[:, :, 2].max()
        _, pred_mask = renderer(
            mode="ortho",
            vertices=verts,
            faces=face,
            rast_size=H,
            orig_size=(W, H),
            anti_aliasing=True,
            scale=scale,
            trans=trans,
            far=far,
        )
        pred_mask = pred_mask.unsqueeze(1)
        pred_mask = F.interpolate(pred_mask, size=(H, W))
        pred_mask = pred_mask.repeat(1, 3, 1, 1)  # (B, C, H, W)
        pred_mask = bchw_2_bhwc(pred_mask).squeeze()
        mask = pred_mask.detach().cpu().numpy() * 255
        mask = mask.astype(np.uint8)  # (H, W, C)
        mask_binary = mask[:, :, 0]

        if args.viz:  # visualize, slow
            image = cv2.addWeighted(image, 0.5, mask, 0.5, 0)
            image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite("tmp/img.png", image)

        if os.path.exists(mask_dump_path):
            continue
        else:
            imageio.imwrite(mask_dump_path, mask_binary)
            bar.set_description(f"rendered to ...{mask_dump_path[-60:]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-sp", "--split", type=str, default="val", help="data split type")
    parser.add_argument("-v", "--viz", action="store_true", help="visualize mode")
    args = parser.parse_args()

    main(args)
