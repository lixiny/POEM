import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
from lib.datasets.dexycb import DexYCBMultiView
from lib.datasets.ho3d import HO3Dv3MultiView
from lib.datasets.oakink import OakInkMultiView
from lib.utils.config import CN
from lib.utils.transform import SE3_transform, denormalize, persp_project
from lib.viztools.draw import imdesc, plot_hand
from lib.utils.points_utils import index_points
from lib.viztools.viz_o3d_utils import VizContext, create_coord_system_can
from manotorch.manolayer import ManoLayer
from termcolor import colored, cprint
from pytorch3d.ops import ball_query

DEXYCB_3D_CONFIG = dict(
    DATA_MODE="3D",
    DATA_ROOT="data",
    DATA_SPLIT="train",
    N_VIEWS=8,
    SETUP="s0",
    USE_LEFT_HAND=False,
    FILTER_INVISIBLE_HAND=True,
    MASTER_SYSTEM="as_constant_camera",
    TRANSFORM=dict(
        TYPE="SimpleTransform3DMultiView",
        AUG=True,
        CENTER_JIT=0.05,
        SCALE_JIT=0.06,
        COLOR_JIT=0.3,
        ROT_JIT=10,
        ROT_PROB=1.0,
        OCCLUSION=True,
        OCCLUSION_PROB=0.2,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=2.0,
        IMAGE_SIZE=(256, 256),
        CENTER_IDX=0,
    ),
)
HO3D_3D_CONFIG = dict(
    DATA_MODE="3D",
    DATA_ROOT="data",
    DATA_SPLIT="train",
    CONST_CAM_ID=2,
    N_VIEWS=5,
    USE_GT_FROM_MULTIVIEW=True,
    SPLIT_MODE="paper",
    ADD_EVALSET_TRAIN=True,
    FILTER_INVISIBLE_HAND=True,
    MASTER_SYSTEM="as_constant_camera",
    TRANSFORM=dict(
        TYPE="SimpleTransform3DMultiView",
        AUG=True,
        CENTER_JIT=0.05,
        SCALE_JIT=0.06,
        COLOR_JIT=0.3,
        ROT_JIT=10,
        ROT_PROB=1.0,
        OCCLUSION=False,
        OCCLUSION_PROB=0.0,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=2.0,
        IMAGE_SIZE=(256, 256),
        CENTER_IDX=0,
    ),
)
OAKINK_3D_CONFIG = dict(
    DATA_MODE="3D",
    DATA_ROOT="data",
    DATA_SPLIT="train+val",
    SPLIT_MODE="subject",
    USE_SPLIT_MV=True,
    USE_PACK=True,
    N_VIEWS=4,
    MASTER_SYSTEM="as_constant_camera",
    TRANSFORM=dict(
        TYPE="SimpleTransform3DMultiView",
        AUG=True,
        CENTER_JIT=0.05,
        SCALE_JIT=0.06,
        COLOR_JIT=0.3,
        ROT_JIT=10,
        ROT_PROB=1.0,
        OCCLUSION=False,
        OCCLUSION_PROB=0.0,
    ),
    DATA_PRESET=dict(
        USE_CACHE=True,
        BBOX_EXPAND_RATIO=2.0,
        IMAGE_SIZE=(256, 256),
        CENTER_IDX=0,
    ),
)


def get_coords(args, img_metas, LID=True):
    eps = 1e-5
    inp_img_h, inp_img_w = img_metas['inp_img_shape']
    #B, N, C, H, W = img_feats[1].shape
    N = img_metas["cam_intr"].shape[0]
    B, H, W = 1, 32, 32
    coords_h = torch.arange(H).float() * inp_img_h / H  # U
    coords_w = torch.arange(W).float() * inp_img_w / W  # V
    depth_num = 32
    if args.dataset == "dexycb":
        depth_start = 0.2
        depth_end = 2.0
    elif args.dataset == "ho3d":
        depth_start = 0.0
        depth_end = 1.2
    elif args.dataset == "oakink":
        depth_start = 0.0
        depth_end = 1.5

    if LID:
        index = torch.arange(start=0, end=depth_num, step=1).float()
        index_plus1 = index + 1
        bin_size = (depth_end - depth_start) / (depth_num * (1 + depth_num))
        coords_d = depth_start + bin_size * index * index_plus1
    else:
        index = torch.arange(start=0, end=depth_num, step=1).float()
        bin_size = (depth_end - depth_start) / depth_num
        coords_d = depth_start + bin_size * index

    D = coords_d.shape[0]

    # (W, H, D, 3)
    coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3, 0)

    # ===== 1. using camera intrinsic to convert UVD 2 XYZ  >>>>>>
    INTR = img_metas["cam_intr"]  # (B, N, 3, 3)
    fx = INTR[..., 0, 0].unsqueeze(dim=-1)  # (B, N, 1)
    fy = INTR[..., 1, 1].unsqueeze(dim=-1)  # (B, N, 1)
    cx = INTR[..., 0, 2].unsqueeze(dim=-1)  # (B, N, 1)
    cy = INTR[..., 1, 2].unsqueeze(dim=-1)  # (B, N, 1)
    cam_param = torch.cat((fx, fy, cx, cy), dim=-1)  # (B, N, 4)
    cam_param = cam_param.view(B, N, 1, 1, 1, 4).repeat(1, 1, W, H, D, 1)  # (B, N, W, H, D, 4)

    coords_uv, coords_d = coords[..., :2], coords[..., 2:3]  # (W, H, D, 2), (W, H, D, 1)
    coords_uv = coords_uv.view(1, 1, W, H, D, 2).repeat(B, N, 1, 1, 1, 1)  # (B, N, W, H, D, 2)
    coords_d = coords_d.view(1, 1, W, H, D, 1).repeat(B, N, 1, 1, 1, 1)  # (B, N, W, H, D, 1)

    coords_uv = (coords_uv - cam_param[..., 2:4]) / cam_param[..., :2]  # (B, N, W, H, D, 2)
    coords_xy = coords_uv * coords_d
    coords_z = coords_d
    coords = torch.cat((coords_xy, coords_z), dim=-1)  # (B, N, W, H, D, 3)

    # ===== 2. using camera extrinsic to transfer childs' XYZ 2 parent's space >>>>>>
    EXTR = img_metas["cam_extr"]  # (B, N, 4, 4)
    coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
    coords = coords.unsqueeze(-1)  # (B, N, W, H, D, 4, 1)
    EXTR = EXTR.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)  # (B, N, W, H, D, 4, 4)
    coords3d = torch.matmul(EXTR, coords).squeeze(-1)[..., :3]  # (B, N, W, H, D, 3),  xyz in parent's space

    return coords3d


def sample_points_from_original_ball(pt_xyz, center_point, k, radius):
    _, ball_idx, xyz = ball_query(center_point, pt_xyz, K=k, radius=radius, return_nn=True)
    # print(ball_idx.shape, xyz.shape)  # (B=1, 1, 2048), (B=1, 1, 2048, 3)
    xyz = xyz.squeeze(1)
    return xyz, ball_idx


def main(args):
    if args.dataset == "dexycb":
        cfg = CN(DEXYCB_3D_CONFIG)
        dataset = DexYCBMultiView(cfg)
    elif args.dataset == "ho3d":
        cfg = CN(HO3D_3D_CONFIG)
        dataset = HO3Dv3MultiView(cfg)
    elif args.dataset == "oakink":
        cfg = CN(OAKINK_3D_CONFIG)
        dataset = OakInkMultiView(cfg)
    else:
        raise ValueError(f"{args.dataset} is not supported")

    cfg.MASTER_SYSTEM = args.master_system
    cprint("viz {}MultiView with master system: {}".format(args.dataset, cfg.MASTER_SYSTEM), "red", on_color="on_white")

    hand_face = ManoLayer(mano_assets_root='assets/mano_v1_2').th_faces.numpy()

    viz_ctx = VizContext(non_block=True)
    viz_ctx.init()
    geometry_to_viz = dict(
        hand_mesh=None,
        master_system=None,
        coord_system_list=None,
    )

    for i in range(len(dataset)):
        i = np.random.randint(len(dataset))
        sample = dataset[i]

        master_id = sample["master_id"]
        if args.dataset != "oakink":
            master_serial = sample["master_serial"]
        else:
            assert dataset.master_system == "as_constant_camera", "OakInkMultiView only support as_constant_camera"

        if dataset.master_system == "as_first_camera":
            assert master_id == 0, "master_id should be 0"
        elif dataset.master_system == "as_constant_camera":
            if args.dataset == "dexycb":
                assert master_serial == dataset.CONST_CAM_SERIAL, "master_serial should be equal to dataset's CONST_CAM_SERIAL"
            if args.dataset == "ho3d":
                assert master_serial == sample["cam_serial"][
                    0][:-1] + f"{dataset.const_cam_id}", "master_serial should be equal to the set serial"

        master_joints_3d = sample["master_joints_3d"]
        master_verts_3d = sample["master_verts_3d"]

        transferred_joints_3d = []
        transferred_verts_2d = []
        transferred_joints_2d = []
        transfom_T_m2c = []
        ori_joints_2d = []
        multiview_images = []

        for j in range(len(sample["sample_idx"])):

            curr_T_m2c = sample["target_cam_extr"][j]
            transfom_T_m2c.append(curr_T_m2c)
            curr_T_c2m = np.linalg.inv(curr_T_m2c)
            curr_cam_intr = sample["target_cam_intr"][j]
            # R = curr_T_c2m[:3, :3]  # (3, 3)
            # t = curr_T_c2m[:3, 3]  # (3)
            joints_3d_trasf_from_master = SE3_transform(master_joints_3d, curr_T_c2m)
            verts_3d_trasf_from_master = SE3_transform(master_verts_3d, curr_T_c2m)
            joints_2d_proj_from_master = persp_project(joints_3d_trasf_from_master, curr_cam_intr)
            verts_2d_proj_from_master = persp_project(verts_3d_trasf_from_master, curr_cam_intr)

            transferred_joints_3d.append(joints_3d_trasf_from_master.copy())
            transferred_joints_2d.append(joints_2d_proj_from_master.copy())
            transferred_verts_2d.append(verts_2d_proj_from_master.copy())

            image = sample["image"][j]
            image = torch.from_numpy(image)
            image = denormalize(image, [0.5, 0.5, 0.5], [1, 1, 1]).numpy().transpose(1, 2, 0)
            image = (image * 255.0).astype(np.uint8).copy()
            multiview_images.append(image)

            curr_joints_3d = sample["target_joints_3d"][j]
            curr_joins_2d = persp_project(curr_joints_3d, curr_cam_intr)
            ori_joints_2d.append(curr_joins_2d.copy())

        # prepare image to show
        proj_img_toshow = []
        gt_img_toshow = []
        for i in range(len(transferred_joints_3d)):
            J3d = transferred_joints_3d[i]
            J2d = transferred_joints_2d[i]
            V2d = transferred_verts_2d[i]
            oriJ2d = ori_joints_2d[i]
            res_proj = plot_hand(multiview_images[i].copy(), J2d, linewidth=2)
            # for _, v2d in enumerate(V2d):
            #     cv2.circle(res_proj, (int(v2d[0]), int(v2d[1])), 2, (0, 0, 255), -1)

            res_ori = plot_hand(multiview_images[i].copy(), oriJ2d, linewidth=2)

            res_proj = imdesc(res_proj, desc=f"{i} (from master J proj)" if i != master_id else f"MASTER PROJ")
            proj_img_toshow.append(res_proj)
            res_ori = imdesc(res_ori, desc=f"{i} (from ori GT)" if i != master_id else f"MASTER GT")
            gt_img_toshow.append(res_ori)

        final_img_to_show = np.vstack([np.hstack(proj_img_toshow), np.hstack(gt_img_toshow)])

        # region ===== prepare 3d to show >>>>>
        img_metas = {}
        img_metas['inp_img_shape'] = torch.tensor(sample["image"].shape[-2:])
        img_metas["cam_intr"] = torch.tensor(sample["target_cam_intr"]).to(torch.float32)
        img_metas["cam_extr"] = torch.tensor(sample["target_cam_extr"]).to(torch.float32)
        points = torch.tensor(sample["master_verts_3d"]).to(torch.float32)
        center_point = torch.mean(points, dim=0, keepdim=True)  # (1, 3)
        coords3d = get_coords(args, img_metas=img_metas, LID=True)  ## # (B, N, W, H, D, 3),
        # coords3d = coords3d[:, 0:2, ...]
        batch_size = coords3d.shape[0]
        n_cams = coords3d.shape[1]
        #print('nc', n_cams)
        colors = torch.ones_like(coords3d.reshape(batch_size, n_cams, -1, 3))
        n_points_percams = colors.shape[2]
        n_points = n_points_percams * n_cams
        #print(n_points)

        for i in range(n_cams):
            colors[:, i:(i + 1), ...] = torch.rand((3)).reshape(1, 1, 1, 3).repeat(batch_size, 1, n_points_percams, 1)

        colors = colors.reshape(batch_size, -1, 3)
        center_point = center_point.unsqueeze(0)  # (B=1, 1, 3)
        alist = torch.randperm(n_points)
        coords3d = coords3d.reshape(batch_size, -1, 3)
        coords3d = coords3d[:, alist, :]
        colors = colors[:, alist, :]

        coords3d, coords_idx = sample_points_from_original_ball(coords3d, center_point, 2048, 0.2)  # (B, N, 3)
        colors = index_points(colors, coords_idx)

        if geometry_to_viz.get("ball_pc", None) is None:
            # grid_cloud = o3d.geometry.PointCloud()
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(coords3d.reshape(-1, 3).cpu().numpy())
            pc.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3).cpu().numpy())
            viz_ctx.add_geometry(pc)
            geometry_to_viz["ball_pc"] = pc
        else:
            pc.points = o3d.utility.Vector3dVector(coords3d.reshape(-1, 3).cpu().numpy())
            pc.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3).cpu().numpy())
            viz_ctx.update_geometry(pc)

        if geometry_to_viz["hand_mesh"] is None:
            hand_mesh = o3d.geometry.TriangleMesh()
            hand_mesh.triangles = o3d.utility.Vector3iVector(hand_face)
            hand_mesh.vertices = o3d.utility.Vector3dVector(master_verts_3d)
            hand_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array([
                [0.8, 0.8, 0.8],
            ] * len(master_verts_3d)))
            hand_mesh.compute_vertex_normals()
            #viz_ctx.add_geometry_list([hand_mesh])
            viz_ctx.add_geometry_list([hand_mesh, pc])
            geometry_to_viz["hand_mesh"] = hand_mesh
        else:
            hand_mesh.vertices = o3d.utility.Vector3dVector(master_verts_3d)
            hand_mesh.compute_vertex_normals()
            viz_ctx.update_geometry(hand_mesh)

        if geometry_to_viz["master_system"] is not None:
            viz_ctx.remove_geometry_list(geometry_to_viz["master_system"])

        master_coord_sys = create_coord_system_can(scale=5)
        viz_ctx.add_geometry_list(master_coord_sys)
        geometry_to_viz["master_system"] = master_coord_sys

        n_Cameras = dataset.n_views
        if geometry_to_viz["coord_system_list"] is not None:
            viz_ctx.remove_geometry_list(geometry_to_viz["coord_system_list"])

        coord_system_list = []
        for i in range(n_Cameras):
            coord_system_list += create_coord_system_can(scale=1, transf=transfom_T_m2c[i])
        viz_ctx.add_geometry_list(coord_system_list)
        geometry_to_viz["coord_system_list"] = coord_system_list
        # endregion

        while True:
            cv2.imshow(
                "tmp/test_dexycb_multiview.png",
                cv2.cvtColor(final_img_to_show, cv2.COLOR_BGR2RGB),
            )
            if viz_ctx is not None:
                viz_ctx.step()
            key = cv2.waitKey(10)
            if key == ord("a") or key == ord("d"):
                break
            elif key == ord("\r") or key == ord("\n"):
                break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="dexycb", choices=["dexycb", "ho3d", "oakink"])
    parser.add_argument("-ms",
                        "--master_system",
                        type=str,
                        default="as_constant_camera",
                        choices=["as_first_camera", "as_constant_camera"])
    parser.add_argument("-v", "--viz", action="store_true", help="visualize mode")
    args = parser.parse_args()

    main(args)