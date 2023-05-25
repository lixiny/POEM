import numpy as np
import torch
from ..utils.builder import build_dataset
from ..utils.logger import logger
from termcolor import colored
from lib.utils.config import CN


class MixDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_cfg_list: list, max_len: int = None, data_preset: CN = None):
        self.datasets = [build_dataset(cfg, data_preset=data_preset) for cfg in dataset_cfg_list]
        self.ratios = np.array([cfg.MIX_RATIO for cfg in dataset_cfg_list])
        self.N_DS = len(self.datasets)
        self.partitions = self.ratios.cumsum()

        assert self.partitions[-1] == 1.0, "Mixing ratios must sum to 1.0"

        self.total_length = sum([len(d) for d in self.datasets])
        self.length = max([len(d) for d in self.datasets])

        if max_len is not None:
            self.length = min(self.length, max_len)

        logger.warning(f"MixDataset initialized Done! "
                       f"Including {len(self.datasets)} datasets and {self.length} working length")

        info = colored(" + ", 'blue', attrs=['bold']).\
            join([f"{self.ratios[i]} * {self.datasets[i].name}" for i in range(self.N_DS)])
        logger.info(f"MixDataset: {info}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Index an element from the dataset.
        This is done by randomly choosing a dataset using the mixing percentages
        and then randomly choosing from the selected dataset.
        Returns:
            Dict: Dictionary containing data and labels for the selected example
        """
        p = np.random.rand()
        for i in range(len(self.datasets)):  # N datasets
            if p <= self.partitions[i]:
                p = np.random.randint(len(self.datasets[i]))
                return self.datasets[i][p]
