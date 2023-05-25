import argparse
import json
from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from ..utils.logger import logger
from scipy.linalg import orthogonal_procrustes

from .basic_metric import AverageMeter, Metric


class PAEval(Metric):

    def __init__(self, cfg, mesh_score=False) -> None:
        super().__init__()

        self.mesh_score = mesh_score

        self.pa_mpjpe = AverageMeter()
        self.mpjpe = AverageMeter()

        if self.mesh_score is True:
            self.pa_mpvpe = AverageMeter()
            self.mpvpe = AverageMeter()

        self.reset()

    def reset(self):

        self.pa_mpjpe.reset()
        self.mpjpe.reset()

        if self.mesh_score is True:
            self.pa_mpvpe.reset()
            self.mpvpe.reset()

    def get_dist(self, x, y):
        diff = x - y  # (B, N, 3)
        dist = np.linalg.norm(diff, axis=2)  # (B, N)
        return np.mean(dist, axis=1)  # (B)

    def feed(self, pred_joints_3d_abs, joints_3d_abs, pred_verts_3d_abs=None, verts_3d_abs=None, **kwargs):

        batch_size = pred_joints_3d_abs.shape[0]
        pred_joints_3d_abs = pred_joints_3d_abs.detach().cpu().numpy()
        joints_3d_abs = joints_3d_abs.cpu().numpy()

        pred_joints_3d_aligned = []
        pred_trafo = []

        # scipy do not support the batch operation
        for i in range(pred_joints_3d_abs.shape[0]):
            pred_joints_3d_aligned_item, trafo_item = self.align_w_scale(joints_3d_abs[i], pred_joints_3d_abs[i])
            pred_joints_3d_aligned.append(pred_joints_3d_aligned_item)
            pred_trafo.append(trafo_item)

        pred_joints_3d_aligned = np.stack(pred_joints_3d_aligned, axis=0)

        pa_mpjpe = self.get_dist(pred_joints_3d_aligned, joints_3d_abs)
        self.pa_mpjpe.update(np.sum(pa_mpjpe), batch_size)
        mpjpe = self.get_dist(pred_joints_3d_abs, joints_3d_abs)
        self.mpjpe.update(np.sum(mpjpe), batch_size)

        if self.mesh_score is True:

            pred_verts_3d_abs = pred_verts_3d_abs.detach().cpu().numpy()
            verts_3d_abs = verts_3d_abs.cpu().numpy()

            pred_verts_3d_aligned = []
            for i in range(pred_verts_3d_abs.shape[0]):
                pred_verts_3d_aligned_item, _ = self.align_w_scale(verts_3d_abs[i], pred_verts_3d_abs[i])
                pred_verts_3d_aligned.append(pred_verts_3d_aligned_item)

            pred_verts_3d_aligned = np.stack(pred_verts_3d_aligned, axis=0)

            pa_mpvpe = self.get_dist(pred_verts_3d_aligned, verts_3d_abs)
            self.pa_mpvpe.update(np.sum(pa_mpvpe), batch_size)
            mpvpe = self.get_dist(pred_verts_3d_abs, verts_3d_abs)
            self.mpvpe.update(np.sum(mpvpe), batch_size)

    def get_measures(self, **kwargs) -> Dict[str, float]:
        measures = {}
        measures["pa_mpjpe"] = self.pa_mpjpe.avg
        measures["mpjpe"] = self.mpjpe.avg
        if self.mesh_score:
            measures["pa_mpvpe"] = self.pa_mpvpe.avg
            measures["mpvpe"] = self.mpvpe.avg
        return measures

    def get_result(self):
        return self.pa_mpjpe.avg

    def __str__(self) -> str:
        score = f"pa_mpjpe(mm): {self.pa_mpjpe.avg * 1000.0 :6.4f} | mpjpe: {self.mpjpe.avg:6.4f}"
        if self.mesh_score:
            score += f" | pa_mpvpe(mm): {self.pa_mpvpe.avg * 1000.0:6.4f} | mpvpe: {self.mpvpe.avg:6.4f}"

        return score

    @staticmethod
    def align_w_scale(mtx1, mtx2):
        """Align the predicted entity in some optimality sense with the ground truth."""
        # center
        t1 = mtx1.mean(0)
        t2 = mtx2.mean(0)
        mtx1_t = mtx1 - t1
        mtx2_t = mtx2 - t2

        # scale
        s1 = np.linalg.norm(mtx1_t) + 1e-8
        mtx1_t /= s1
        s2 = np.linalg.norm(mtx2_t) + 1e-8
        mtx2_t /= s2

        # orth alignment
        R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

        # apply trafos to the second matrix
        mtx2_t = np.dot(mtx2_t, R.T) * s
        mtx2_t = mtx2_t * s1 + t1
        return mtx2_t, (R, s, s1, t1 - t2)
