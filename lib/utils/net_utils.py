import os
import random
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad
from torch.optim import Optimizer

from .logger import logger
from .misc import CONST


def worker_init_fn(worker_id):
    seed = worker_id * int(torch.initial_seed()) % CONST.INT_MAX
    np.random.seed(seed)
    random.seed(seed)


def freeze_batchnorm_stats(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        freeze_batchnorm_stats(child)


def recurse_freeze(model):
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = 0
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        recurse_freeze(child)


def build_optimizer(params: Iterable, cfg) -> Optimizer:
    if cfg.OPTIMIZER in ["Adam", "adam"]:
        return torch.optim.Adam(
            params,
            lr=cfg.LR,
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    elif cfg.OPTIMIZER == "AdamW":
        return torch.optim.AdamW(
            params,
            lr=cfg.LR,
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.01)),
        )

    elif cfg.OPTIMIZER in ["SGD", "sgd"]:
        return torch.optim.SGD(
            params,
            lr=cfg.LR,
            momentum=float(cfg.get("MOMENTUM", 0.0)),
            weight_decay=float(cfg.get("WEIGHT_DECAY", 0.0)),
        )
    else:
        raise NotImplementedError(f"{cfg.OPTIMIZER} not be implemented yet")


def build_scheduler(optimizer: Optimizer, cfg):
    scheduler = cfg.SCHEDULER
    lr_decay_step = cfg.get("LR_DECAY_STEP", -1)
    tar_scheduler = None

    if isinstance(lr_decay_step, list) and scheduler == "StepLR":
        tar_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )

    elif scheduler == "StepLR":
        tar_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )

    elif scheduler == "MultiStepLR":
        tar_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.LR_DECAY_STEP,
            gamma=cfg.LR_DECAY_GAMMA,
        )
    else:
        raise NotImplementedError(f"{scheduler} not yet be implemented")

    return tar_scheduler


def clip_gradient(optimizer, max_norm, norm_type):
    """Clips gradients computed during backpropagation to avoid explosion of gradients.

    Args:
        optimizer (torch.optim.optimizer): optimizer with the gradients to be clipped
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            clip_grad.clip_grad_norm_(param, max_norm, norm_type)


def setup_seed(seed, conv_repeatable=True):
    """Setup all the random seeds

    Args:
        seed (int or float): seed value
        conv_repeatable (bool, optional): Whether the conv ops are repeatable (depend on cudnn). Defaults to True.
    """
    logger.warning(f"setup random seed : {seed}")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if conv_repeatable:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        logger.warning("Exp result NOT repeatable!")


### Initialize module parameters with values according to the method ###
def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def xavier_init(module, gain=1, bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def uniform_init(module, a=0, b=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.uniform_(module.weight, a, b)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def init_weights(moudle: nn.Module, pretrained=None):
    if pretrained == "" or pretrained is None:
        logger.warning(f"=> Init {type(moudle).__name__} weights in its' backbone and head")
        """
        Add init for other modules
        ...
        """
    elif os.path.isfile(pretrained):
        logger.info(f"=> Loading {type(moudle).__name__} pretrained model from: {pretrained}")
        # self.load_state_dict(pretrained_state_dict, strict=False)
        checkpoint = torch.load(pretrained, map_location=torch.device("cpu"))
        if isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict_old = checkpoint["state_dict"]
            state_dict = OrderedDict()
            # delete 'module.' because it is saved from DataParallel module
            for key in state_dict_old.keys():
                if key.startswith("module."):
                    # state_dict[key[7:]] = state_dict[key]
                    # state_dict.pop(key)
                    state_dict[key[7:]] = state_dict_old[key]  # delete "module." (in nn.parallel)
                else:
                    state_dict[key] = state_dict_old[key]
        else:
            logger.error(f"=> No state_dict found in checkpoint file {pretrained}")
            raise RuntimeError()
        moudle.load_state_dict(state_dict, strict=False)
        logger.info(f"=> Loading SUCCEEDED")
    else:
        logger.error(f"=> No {type(moudle).__name__} checkpoints file found in {pretrained}")
        raise FileNotFoundError()
