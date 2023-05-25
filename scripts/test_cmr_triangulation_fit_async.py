import math
import os
import pickle
from pprint import pformat
from time import time

import numpy as np
import torch
from joblib import Parallel, delayed
from lib.external.cmr.data_adaptor import map2uv
from lib.fit.frame_fit.one_frame_fit_silh import OneFrameFitSilh
from lib.metrics import MeanEPE, PAEval
from lib.opt import merge_args, parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_perfixes, format_args_cfg
from lib.utils.net_utils import setup_seed, worker_init_fn
from lib.utils.transform import batch_cam_extr_transf
from lib.metrics.pck import Joint3DPCK, Vert3DPCK
from manotorch.manolayer import ManoLayer
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

fmt = "{desc:<5}{percentage:3.0f}%|{bar:60}{r_bar}"


def evaluate_triangulation(cfg: CN, eval_load_dir: str, eval_dump_path: str):
    all_fitted = [f for f in os.listdir(eval_load_dir) if f.endswith('.pkl')]

    MPJPE_CS = MeanEPE(cfg=None, name="J_cs")
    MPVPE_CS = MeanEPE(cfg=None, name="V_cs")
    MPJPE = MeanEPE(cfg=None, name="J")
    MPVPE = MeanEPE(cfg=None, name="V")
    PA = PAEval(cfg=None, mesh_score=True)
    center_idx = cfg.DATA_PRESET.CENTER_IDX
    PCK_J = Joint3DPCK(EVAL_TYPE="joints_3d", VAL_MIN=0.0, VAL_MAX=0.02, STEPS=20)
    PCK_V = Vert3DPCK(EVAL_TYPE="verts_3d", VAL_MIN=0.0, VAL_MAX=0.02, STEPS=20)

    pbar = tqdm(total=len(all_fitted), position=0, bar_format=fmt, desc="fit: ")
    for elm in all_fitted:
        sample_name = elm
        sample_path = os.path.join(eval_load_dir, sample_name)
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        gt_joints = sample["gt_joints"].squeeze(0)
        gt_verts = sample["gt_verts"].squeeze(0)
        fitted_joints = sample["fitted_joints"]
        fitted_verts = sample["fitted_verts"]

        preds = dict()
        targs = dict()
        preds["pred_joints_3d"] = sample["fitted_joints"]
        preds["pred_verts_3d"] = sample["fitted_verts"]
        targs["master_joints_3d"] = sample["gt_joints"].squeeze(0)
        targs["master_verts_3d"] = sample["gt_verts"].squeeze(0)
        PCK_J.feed(preds, targs)
        PCK_V.feed(preds, targs)

        MPJPE_CS.feed(fitted_joints, gt_joints)
        MPVPE_CS.feed(fitted_verts, gt_verts)
        PA.feed(fitted_joints, gt_joints, fitted_verts, gt_verts)

        fitted_joints_rel = fitted_joints - fitted_joints[:, center_idx].unsqueeze(1)
        gt_joints_rel = gt_joints - gt_joints[:, center_idx].unsqueeze(1)
        MPJPE.feed(fitted_joints_rel, gt_joints_rel)

        fitted_verts_rel = fitted_verts - fitted_joints[:, center_idx].unsqueeze(1)
        gt_verts_rel = gt_verts - gt_joints[:, center_idx].unsqueeze(1)
        MPVPE.feed(fitted_verts_rel, gt_verts_rel)

        pbar.set_description(
            f"eval: JCS={MPJPE_CS.get_result():.5f} | VCS={MPVPE_CS.get_result():.5f} | PAJ={PA.get_result():.5f}")
        pbar.update(1)

    metric_list = [MPJPE_CS, MPJPE, MPVPE_CS, MPVPE, PA, PCK_J, PCK_V]
    with open(eval_dump_path, "a") as f:
        for M in metric_list:
            f.write(f"{pformat(M.get_measures())}\n")
        f.write("\n")


def fit_worker(
    device: torch.device,
    worker_id: int,
    n_workers: int,
    fit_load_dir: str,
    fit_dump_dir: str,
    mano_center_idx: int,
    image_size: int = 256,
    silh_size: int = 128,
    lr: float = 1e-2,
    grad_clip: float = 1e-1,
):
    all_samples = [f for f in os.listdir(fit_load_dir) if f.endswith('.pkl')]
    begin_index = worker_id * len(all_samples) // n_workers
    end_index = (worker_id + 1) * len(all_samples) // n_workers
    n_samples = end_index - begin_index

    cprint(f"====== {worker_id:>3} begin: {begin_index:0>4} end: {end_index:0>4} len: {n_samples} >>>>>>", "cyan")
    cprint(f"====== {worker_id:>3} using device: {device} >>>>>>", "cyan")

    mano_layer = ManoLayer(
        rot_mode="quat",
        use_pca=False,
        side="right",
        mano_assets_root="assets/mano_v1_2",
        center_idx=mano_center_idx,
        flat_hand_mean=True,
    ).to(device)

    fitting_unit = OneFrameFitSilh(
        device=device,
        lr=lr,
        grad_clip=grad_clip,
        side="right",
        mano_layer=mano_layer,
        image_size=image_size,
        silh_size=silh_size,
        lambda_reprojection_loss=1000.0,
        lambda_anatomical_loss=50.0,
        lambda_mask_render_loss=10.0,
        gamma_joint_b_axis_loss=1.0,
        gamma_joint_u_axis_loss=0.1,
        gamma_joint_l_limit_loss=0.1,
        gamma_angle_limit_loss=0.1,
    )

    cprint(f"====== fitting unit created on device: {device} >>>>>>", "cyan")

    res_list = []
    for i in range(begin_index, end_index):
        sample_name = f"{i}.pkl"
        sample_path = os.path.join(fit_load_dir, sample_name)
        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        # already finished this sampled
        if os.path.exists(os.path.join(fit_dump_dir, f"{i}.pkl")):
            continue

        n_persp = sample["keypoint_arr"].shape[0]
        init_pose = np.array([[1.0, 0.0, 0.0, 0.0]] * 16).astype(np.float32)  # (16, 4)
        init_pose = torch.from_numpy(init_pose.reshape(1, 64))  # (1, 64)
        # init_pose add small disturbance (avoid Loss nan in new manotorch)
        init_pose = init_pose + 0.1 * torch.rand_like(init_pose)  # @lixin
        init_shape = torch.zeros((1, 10), dtype=torch.float)  # (1, 10)

        init_tsl = np.array([0.0, 0.0, 0.6]).astype(np.float32)  # (3)
        init_tsl = torch.from_numpy(init_tsl.reshape(1, 3))  # (1, 3)

        fitting_unit.setup(
            n_iter=300,
            image_scale=sample["image_scale"],
            enable_silh_loss=True,
            const_keypoint_arr=sample["keypoint_arr"],
            const_mask_arr=sample["mask_pred"],
            const_cam_intr_arr=sample["cam_intr"],
            const_cam_extr_arr=sample["cam_extr"],
            init_hand_pose=init_pose,
            init_hand_shape=init_shape,
            init_hand_tsl=init_tsl,
        )

        init_proj_loss, final_proj_loss = 0, 0
        while True:
            optimize_condition = fitting_unit.condition()
            curr_iter = fitting_unit.curr_iter
            loop_condition = optimize_condition
            if not loop_condition:
                break

            if optimize_condition:
                proj_loss_val, _ = fitting_unit.step()
                if curr_iter == 1:
                    init_proj_loss = proj_loss_val

        final_proj_loss = proj_loss_val
        fitted_joints, fitted_verts = fitting_unit.recover_hand(squeeze_out=False)  # (B=1, NJOINT, 3)

        fitted_joints = fitted_joints.detach().cpu()  # (B=1, 21, 3)
        fitted_joints_multicam = fitted_joints.unsqueeze(1).expand(-1, n_persp, -1, -1)  # (B=1, NPERSP, 21, 3)
        fitted_joints_multicam = batch_cam_extr_transf(sample["cam_extr"].clone()[None, ...], fitted_joints_multicam)
        fitted_joints_multicam = fitted_joints_multicam.squeeze(0)  # (NPERSP, 21, 3)
        fitted_verts = fitted_verts.detach().cpu()  # (B=1, 778, 3)
        fitted_verts_multicam = fitted_verts.unsqueeze(1).expand(-1, n_persp, -1, -1)  # (B=1, NPERSP, 778, 3)
        fitted_verts_multicam = batch_cam_extr_transf(sample["cam_extr"].clone()[None, ...], fitted_verts_multicam)
        fitted_verts_multicam = fitted_verts_multicam.squeeze(0)  # (NPERSP, 778, 3)

        res = {
            "sample_idx": i,
            "gt_joints": sample["gt_joints"],  # (1, NPERSP, 21, 3)
            "fitted_joints": fitted_joints_multicam.clone(),  # (NPERSP, 21, 3)
            "gt_verts": sample["gt_verts"],  # (1, NPERSP, 778, 3)
            "fitted_verts": fitted_verts_multicam.clone(),
        }
        with open(os.path.join(fit_dump_dir, f"{i}.pkl"), 'wb') as f:
            pickle.dump(res, f)

        res_list.append(res)
        cprint(f"====== {worker_id:>3} done {i:>4} LOSS bef: {init_proj_loss:.4f}, aft: {final_proj_loss:.4f} >>>>>>",
               "green")
        fitting_unit.reset()

    return res_list


def triangulation_fit(
    cfg: CN,
    fit_load_dir: str,
    fit_dump_dir: str,
    n_workers: int,
):
    os.makedirs(fit_dump_dir, exist_ok=True)
    # get all cuda device ids
    device_count = torch.cuda.device_count()
    logger.info(f"fitting use total {device_count} GPUS")

    # create device for each worker
    device_list = []
    for worker_id in range(n_workers):
        device_list.append(torch.device(f"cuda:{worker_id % device_count}"))

    # initial jobs
    collected = Parallel(n_jobs=n_workers, verbose=50)(delayed(fit_worker)(
        device=device_list[worker_id],
        worker_id=worker_id,
        n_workers=n_workers,
        fit_load_dir=fit_load_dir,
        fit_dump_dir=fit_dump_dir,
        mano_center_idx=cfg.DATA_PRESET.CENTER_IDX,
        image_size=int(cfg.DATA_PRESET.IMAGE_SIZE[0]),
        silh_size=int(cfg.DATA_PRESET.IMAGE_SIZE[0]) // 2,
    ) for worker_id in range(n_workers))

    # print(len(collected))
    # with open(os.path.join(fit_dump_dir, "fitting_result.pkl"), 'wb') as f:
    #     pickle.dump(collected, f)


def cmr_dump_res(
    cfg: CN,
    dump_dir: str,
):
    # if the model is from the external package

    rank = 0  # only one process.
    test_data = builder.build_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=worker_init_fn)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model = model.to(device=rank)
    device = torch.device(f"cuda:{0}")
    os.makedirs(dump_dir, exist_ok=True)

    def dump_res(preds, inputs, step_idx, **kwargs):
        img = inputs['img']  # (NPERSP, 3, H, W)
        V_STD = 0.2
        gt_joints = inputs['xyz_gt'] * V_STD + inputs['xyz_root'].unsqueeze(1)  # (NPERSP, 21, 3)
        gt_joints = gt_joints.unsqueeze(0).detach().cpu()  # (B=1, NPERSP, 21, 3)
        gt_verts = inputs['mesh_gt'] * V_STD + inputs['xyz_root'].unsqueeze(1)  # (1, NPERSP, 21, 3)
        # use diagonal as scale
        image_scale = math.sqrt(float(img.shape[-1]**2 + img.shape[-2]**2))
        uv_pred = preds['uv_pred']  # (NPERSP, 21, 128, 128)
        uv_point_pred, uv_pred_conf = map2uv(uv_pred.detach().cpu().numpy(),
                                             (img.size(2), img.size(3)))  # (NPERSP, 21, 2), (NPERSP, 21, 1)
        keypoint_arr = np.concatenate([uv_point_pred, uv_pred_conf], axis=-1)  # (NPERSP, 21, 3)
        keypoint_arr = torch.from_numpy(keypoint_arr)  # (BNPERSP, 21, 3)

        cam_intr = inputs['K']  # (NPERSP, 3, 3) intrinsic matrix
        cam_extr = inputs['extr']  # (NPERSP, 4, 4) extrinsic matrix
        mask_pred = preds['mask_pred']  # (NPERSP, H:128, W:128)

        samples_res = {
            'image_scale': image_scale,
            'keypoint_arr': keypoint_arr,
            'mask_pred': mask_pred.detach().cpu(),
            'cam_intr': cam_intr.detach().cpu(),
            'cam_extr': cam_extr.detach().cpu(),
            'gt_joints': gt_joints.detach().cpu(),
            'gt_verts': gt_verts.detach().cpu(),
        }
        with open(os.path.join(dump_dir, f"{step_idx}.pkl"), 'wb') as f:
            pickle.dump(samples_res, f)

    model.eval()
    testbar = etqdm(test_loader, rank=rank)
    for bidx, batch in enumerate(testbar):
        for k, v in batch.items():
            if isinstance(v, list):
                batch[k] = v[0]
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            elif isinstance(v, torch.Tensor):
                batch[k] = v.squeeze(0).to(device=rank)

        step_idx = 0 * len(test_loader) + bidx
        model(batch, step_idx, "test", callback=dump_res, with_registration=False)
        testbar.set_description(f"{bar_perfixes['test']} {model.format_metric('test')}")


if __name__ == "__main__":
    exp_time = time()
    import argparse
    cus_parser = argparse.ArgumentParser(description="")
    cus_parser.add_argument("--cmr_dump", action="store_true")
    cus_parser.add_argument("--net_dump_dir", type=str, default="tmp/triangulation/cmr_res")
    cus_parser.add_argument("--fit", action="store_true")
    cus_parser.add_argument("--fit_dump_dir", type=str, default="tmp/triangulation/fit_res")
    cus_parser.add_argument("--eval_dump_path", type=str, default="tmp/triangulation/fit_Metrics.txt")
    cus_parser.add_argument("--n_fit_workers", type=int, default=16)

    arg, custom_arg_string = parse_exp_args()
    cus_arg = cus_parser.parse_args(custom_arg_string)
    arg = merge_args(arg, cus_arg)
    # assert arg.reload is not None, "reload checkpointint path is required"
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)
    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)
    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    logger.info("====> Testing on single GPU (Data Parallel) <====")
    if arg.cmr_dump:
        cmr_dump_res(
            cfg=cfg,
            dump_dir=arg.net_dump_dir,
        )

    if arg.fit:
        triangulation_fit(
            cfg=cfg,
            fit_load_dir=arg.net_dump_dir,
            fit_dump_dir=arg.fit_dump_dir,
            n_workers=arg.n_fit_workers,
        )

    if arg.evaluate:
        evaluate_triangulation(
            cfg=cfg,
            eval_load_dir=arg.fit_dump_dir,
            eval_dump_path=arg.eval_dump_path,
        )
