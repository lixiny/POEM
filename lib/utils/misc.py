import functools
import math
import re
from collections import abc, namedtuple
from enum import Enum

import numpy as np
import yaml
from termcolor import colored

bar_perfixes = {
    "train": colored("train", "white", attrs=["bold"]),
    "val": colored("val", "yellow", attrs=["bold"]),
    "test": colored("test", "magenta", attrs=["bold"]),
}

RandomState = namedtuple(
    "RandomState",
    [
        "torch_rng_state",
        "torch_cuda_rng_state",
        "torch_cuda_rng_state_all",
        "numpy_rng_state",
        "random_rng_state",
    ],
)
RandomState.__new__.__default__ = (None,) * len(RandomState._fields)


def enable_lower_param(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kw_uppers = {}
        for k, v in kwargs.items():
            kw_uppers[k.upper()] = v
        return func(*args, **kw_uppers)

    return wrapper


def singleton(cls):
    _instance = {}

    @functools.wraps(cls)
    def inner(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner


class ImmutableClass(type):

    def __call__(cls, *args, **kwargs):
        raise AttributeError("Cannot instantiate this class")

    def __setattr__(cls, name, value):
        raise AttributeError("Cannot modify immutable class")

    def __delattr__(cls, name):
        raise AttributeError("Cannot delete immutable class")


class CONST(metaclass=ImmutableClass):
    PI = math.pi
    INT_MAX = 2**32 - 1
    NUM_JOINTS = 21
    SIDE = "right"
    UVD_DEPTH_RANGE = 0.4  # m
    JOINTS_IDX_PARENTS = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
    REF_BONE_LEN = 0.09473151311686484  # in meter

    # https://github.com/lmb-freiburg/freihand/blob/master/utils/mano_utils.py
    MANO_KPID_2_VERTICES = {
        4: [744],  #ThumbT
        8: [320],  #IndexT
        12: [443],  #MiddleT
        16: [555],  #RingT
        20: [672]  #PinkT
    }

    YCB_IDX2CLASSES = {
        1: "002_master_chef_can",
        2: "003_cracker_box",
        3: "004_sugar_box",
        4: "005_tomato_soup_can",
        5: "006_mustard_bottle",
        6: "007_tuna_fish_can",
        7: "008_pudding_box",
        8: "009_gelatin_box",
        9: "010_potted_meat_can",
        10: "011_banana",
        11: "019_pitcher_base",
        12: "021_bleach_cleanser",
        13: "024_bowl",
        14: "025_mug",
        15: "035_power_drill",
        16: "036_wood_block",
        17: "037_scissors",
        18: "040_large_marker",
        19: "051_large_clamp",
        20: "052_extra_large_clamp",
        21: "061_foam_brick",
    }


def update_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        return config


def format_cfg(cfg, indent_lvl=0):
    indent_width = 2
    INDENT = ' ' * indent_width

    cfg_str = ""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            cfg_str += f"\n{INDENT * indent_lvl} * {colored(k, 'magenta')}: {format_cfg(v, indent_lvl+1)}"
    elif isinstance(cfg, (list, tuple)):
        for elm in cfg:
            cfg_str += f"\n{INDENT * (indent_lvl)} - {format_cfg(elm, indent_lvl+1)}"
        cfg_str += f"\n"
    else:
        cfg_str += f"{cfg}"
    return cfg_str


def format_args_cfg(args, cfg={}):
    args_list = [f" - {colored(name, 'green')}: {getattr(args, name)}" for name in vars(args)]
    arg_str = "\n".join(args_list)
    cfg_str = format_cfg(cfg)
    return arg_str + cfg_str


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6


def param_size(net):
    # ! treat all parameters to be float
    return sum(p.numel() for p in net.parameters()) * 4 / (1024 * 1024)


def camel_to_snake(camel_input):
    words = re.findall(r'[A-Z]?[a-z]+|[A-Z]{1,}(?=[A-Z][a-z]|\d|\W|$)|\d+', camel_input)
    return '_'.join(map(str.lower, words))


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True
