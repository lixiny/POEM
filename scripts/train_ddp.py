import os
import random
from argparse import Namespace
from time import time

import lib.models
import numpy as np
import torch
from lib.datasets import create_dataset
from lib.external import EXT_PACKAGE
from lib.opt import parse_exp_args
from lib.utils import builder
from lib.utils.config import get_config
from lib.utils.etqdm import etqdm
from lib.utils.logger import logger
from lib.utils.misc import CONST, bar_perfixes, format_args_cfg
from lib.utils.net_utils import build_optimizer, build_scheduler, clip_gradient, setup_seed
from lib.utils.recorder import Recorder
from lib.utils.summary_writer import DDPSummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.utils.config import CN


def _init_fn(worker_id):
    seed = ((worker_id + 1) * int(torch.initial_seed())) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def main_worker(gpu_id: int, cfg: CN, arg: Namespace, time_f: float):

    # if the model is from the external package
    if cfg.MODEL.TYPE in EXT_PACKAGE:
        pkg = EXT_PACKAGE[cfg.MODEL.TYPE]
        exec(f"from lib.external import {pkg}")

    if arg.distributed:
        rank = arg.n_gpus * arg.node_rank + gpu_id
        torch.distributed.init_process_group(arg.dist_backend, rank=rank, world_size=arg.world_size)
        assert rank == torch.distributed.get_rank(), "Something wrong with nodes or gpus"
        torch.cuda.set_device(rank)
    else:
        rank = None  # only one process.

    setup_seed(cfg.TRAIN.MANUAL_SEED + rank, cfg.TRAIN.CONV_REPEATABLE)
    recorder = Recorder(arg.exp_id, cfg, rank=rank, time_f=time_f)
    summary = DDPSummaryWriter(log_dir=recorder.tensorboard_path, rank=rank)
    # summarizer = Summarizer(arg.exp_id, cfg, rank=rank, time_f=time_f)

    # add a barrier, to make sure all recorders are created
    torch.distributed.barrier()

    train_data = create_dataset(cfg.DATASET.TRAIN, data_preset=cfg.DATA_PRESET)
    train_sampler = DistributedSampler(train_data, num_replicas=arg.world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_data,
                              batch_size=arg.batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=int(arg.workers),
                              pin_memory=True,
                              drop_last=True,
                              sampler=train_sampler,
                              worker_init_fn=_init_fn,
                              persistent_workers=True)

    if rank == 0:
        val_data = create_dataset(cfg.DATASET.TEST, data_preset=cfg.DATA_PRESET)
        val_loader = DataLoader(val_data,
                                batch_size=arg.val_batch_size,
                                shuffle=True,
                                num_workers=int(arg.workers),
                                pin_memory=True,
                                drop_last=False,
                                worker_init_fn=_init_fn)
    else:
        val_loader = None

    model = builder.build_model(cfg.MODEL, data_preset=cfg.DATA_PRESET, train=cfg.TRAIN)
    model.setup(summary_writer=summary)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=cfg.TRAIN.FIND_UNUSED_PARAMETERS, static_graph=True)

    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer = build_optimizer(model.parameters(), cfg=cfg.TRAIN)
    scheduler = build_scheduler(optimizer, cfg=cfg.TRAIN)

    if arg.resume:
        epoch = recorder.resume_checkpoints(model, optimizer, scheduler, arg.resume)
    else:
        epoch = 0

    # Make sure model is created, resume is finished
    torch.distributed.barrier()

    logger.warning(f"############## start training from {epoch} to {cfg.TRAIN.EPOCH} ##############")
    for epoch_idx in range(epoch, cfg["TRAIN"]["EPOCH"]):
        if arg.distributed:
            train_sampler.set_epoch(epoch_idx)

        model.train()
        trainbar = etqdm(train_loader, rank=rank)
        for bidx, batch in enumerate(trainbar):
            optimizer.zero_grad()
            step_idx = epoch_idx * len(train_loader) + bidx
            preds, loss_dict = model(batch, step_idx, "train")
            loss = loss_dict["loss"]
            loss.backward()
            if cfg.TRAIN.GRAD_CLIP_ENABLED:
                clip_gradient(optimizer, cfg.TRAIN.GRAD_CLIP.NORM, cfg.TRAIN.GRAD_CLIP.TYPE)

            optimizer.step()
            optimizer.zero_grad()

            trainbar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} "
                                     f"{model.module.format_metric('train')}")

        scheduler.step()
        logger.info(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

        recorder.record_checkpoints(model, optimizer, scheduler, epoch_idx, arg.snapshot)
        torch.distributed.barrier()
        model.module.on_train_finished(recorder, epoch_idx)

        if (epoch_idx % arg.eval_interval == arg.eval_interval - 1 or epoch_idx == 0) and rank == 0:
            logger.info("do validation and save results")
            with torch.no_grad():
                model.eval()
                valbar = etqdm(val_loader, rank=rank)
                for bidx, batch in enumerate(valbar):
                    step_idx = epoch_idx * len(val_loader) + bidx
                    preds = model(batch, step_idx, "val")

            model.module.on_val_finished(recorder, epoch_idx)

    if arg.distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    exp_time = time()
    arg, _ = parse_exp_args()
    if arg.resume:
        logger.warning(f"config will be reloaded from {os.path.join(arg.resume, 'dump_cfg.yaml')}")
        arg.cfg = os.path.join(arg.resume, "dump_cfg.yaml")
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=False)
    else:
        cfg = get_config(config_file=arg.cfg, arg=arg, merge=True)

    os.environ["MASTER_ADDR"] = arg.dist_master_addr
    os.environ["MASTER_PORT"] = arg.dist_master_port
    # must have equal gpus on each node.
    arg.world_size = arg.n_gpus * arg.nodes
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    arg.batch_size = int(arg.batch_size / arg.n_gpus)
    if arg.val_batch_size is None:
        arg.val_batch_size = arg.batch_size

    arg.workers = int((arg.workers + arg.n_gpus - 1) / arg.n_gpus)

    logger.warning(f"final args and cfg: \n{format_args_cfg(arg, cfg)}")
    # input("Confirm (press enter) ?")

    logger.info("====> Use Distributed Data Parallel <====")
    torch.multiprocessing.spawn(main_worker, args=(cfg, arg, exp_time), nprocs=arg.n_gpus)
