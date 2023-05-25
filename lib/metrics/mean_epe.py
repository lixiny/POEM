from abc import ABC, abstractmethod
from typing import Dict

import torch

from ..utils.logger import logger
from ..utils.misc import CONST
from .basic_metric import AverageMeter, Metric


class MeanEPE(Metric):

    def __init__(self, cfg, name="") -> None:
        super(MeanEPE, self).__init__()
        self.cfg = cfg
        self.avg_meter = AverageMeter()
        self.name = f"{name}_mepe"
        self.reset()

    def reset(self):
        self.avg_meter.reset()

    def feed(self, pred_kp, gt_kp, kp_vis=None, **kwargs):
        assert len(pred_kp.shape) == 3, logger.error(
            "X pred shape, should as (BATCH, NPOINTS, 1|2|3)")  # TENSOR (BATCH, NPOINTS, 2|3)

        diff = pred_kp - gt_kp  # TENSOR (B, N, 1|2|3)
        dist_ = torch.norm(diff, p="fro", dim=2)  # TENSOR (B, N)
        dist_batch = torch.mean(dist_, dim=1, keepdim=True)  # TENSOR (B, 1)
        BATCH_SIZE = dist_batch.shape[0]
        sum_dist_batch = torch.sum(dist_batch)
        self.avg_meter.update(sum_dist_batch.item(), n=BATCH_SIZE)
        return sum_dist_batch.item()

    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        avg = self.avg_meter.avg
        measures[f"{self.name}"] = avg
        return measures

    def get_result(self):
        return self.avg_meter.avg

    def __str__(self):
        return f"{self.name}: {self.avg_meter.avg:6.4f}"
