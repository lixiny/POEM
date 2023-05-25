import os
import random
from argparse import Namespace
from time import time

import warnings
import lib.models
import numpy as np
import torch
from lib.datasets import create_dataset
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import CN, get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_perfixes, format_args_cfg
from lib.utils.net_utils import setup_seed
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from lib.utils.testing import IdleCallback, AUCCallback, DrawingHandCallback
from torch.nn.parallel import DataParallel as DP
from torch.utils.data import DataLoader


def _init_fn(worker_id):
    seed = worker_id * int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def main_worker(cfg: CN, arg: Namespace, time_f: float):
    # if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")

    rank = 0  # only one process.

    if arg.exp_id != 'default':
        warnings.warn("You shouldn't assign exp_id in test mode")
    cfg_name = arg.cfg.split("/")[-1].split(".")[0]
    exp_id = f"eval_{cfg_name}"

    recorder = Recorder(exp_id, cfg, rank=rank, time_f=time_f, eval_only=True)
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)
    test_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
    test_loader = DataLoader(test_data,
                             batch_size=arg.batch_size,
                             shuffle=False,
                             num_workers=int(arg.workers),
                             drop_last=False,
                             worker_init_fn=_init_fn)

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=summary)
    model = DP(model).to(device=rank)

    # define the callback, invoked after each batch forward
    if arg.test_extra == "auc":
        val_max = 0.05 if test_data.__class__.__name__ == 'HO3Dv3MultiView' else 0.02
        cb = AUCCallback(val_max=val_max, exp_dir=os.path.join(recorder.eval_dump_path))
    elif arg.test_extra == "draw":
        cb = DrawingHandCallback(img_draw_dir=os.path.join(recorder.dump_path, "draws"))
    else:
        cb = IdleCallback()  # do nothing

    with torch.no_grad():
        model.eval()
        testbar = etqdm(test_loader, rank=rank)
        for bidx, batch in enumerate(testbar):
            if bidx >= 2:
                break
            step_idx = 0 * len(test_loader) + bidx
            preds = model(batch, step_idx, "test", callback=cb)
            testbar.set_description(f"{bar_perfixes['test']} {model.module.format_metric('test')}")

        model.module.on_test_finished(recorder, 0)
        cb.on_finished()  # deal with the callback results


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    assert arg.reload is not None, "reload checkpoint path is required"
    cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    setup_seed(cfg.TRAIN.MANUAL_SEED, cfg.TRAIN.CONV_REPEATABLE)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    # input("Confirm (press enter) ?")

    logger.info("====> Evaluation on single GPU (Data Parallel) <====")
    main_worker(cfg, arg, exp_time)
