from lib.utils.config import CN

from ..utils.builder import build_dataset
from .dexycb import DexYCB, DexYCBMultiView
from .ho3d import HO3D, HO3Dv3MultiView
from .oakink import OakInk
from .mix_dataset import MixDataset




def create_dataset(cfg: CN, data_preset: CN):
    """
    Create a dataset instance.
    """
    if cfg.TYPE == "MixDataset":
        # list of CN of each dataset
        if isinstance(cfg.DATASET_LIST, dict):
            dataset_list = [v for k, v in cfg.DATASET_LIST.items()]
        else:
            dataset_list = cfg.DATASET_LIST
        return MixDataset(dataset_list, cfg.MAX_LEN, data_preset=data_preset)
    else:
        # default building from cfg
        return build_dataset(cfg, data_preset=data_preset)
