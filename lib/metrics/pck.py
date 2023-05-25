from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import torch
from .basic_metric import AverageMeter, Metric
from ..utils.logger import logger
from ..utils.misc import CONST


class _PCKMetric(Metric, ABC):

    def __init__(self, **cfg) -> None:
        super().__init__()
        self.data = list()
        self.eval_type = "_PCK"
        self.val_min = cfg["VAL_MIN"]
        self.val_max = cfg["VAL_MAX"]
        self.steps = cfg["STEPS"]

    @property
    @abstractmethod
    def num_kp(self):
        pass

    @abstractmethod
    def _get_predictions(self, preds: Dict, targs: Dict):
        pass

    def reset(self):
        # allocate measurement entrance
        self.data = list()
        for _ in range(self.num_kp):
            self.data.append(list())
        self.count = 0

    def feed(self, preds: Dict, targs: Dict, **kwargs):
        """
        Used to feed data to the class.
        Stores the euclidean distance between gt and pred, when it is visible.
        """

        kp_preds, kp_targs, kp_vis = self._get_predictions(preds, targs)

        if kp_vis is None:
            kp_vis = torch.ones_like(kp_targs[..., 1])

        if isinstance(kp_preds, torch.Tensor):
            kp_preds = kp_preds.detach().cpu().numpy()
        if isinstance(kp_targs, torch.Tensor):
            kp_targs = kp_targs.detach().cpu().numpy()
        if isinstance(kp_vis, torch.Tensor):
            kp_vis = kp_vis.detach().cpu().numpy()

        kp_preds = np.squeeze(kp_preds)
        kp_targs = np.squeeze(kp_targs)
        kp_vis = np.squeeze(kp_vis).astype("bool")

        assert len(kp_preds.shape) == 3, logger.error("X kp_preds shape")  # ARRAY (BATCH, NPOINTS, 3)
        assert len(kp_targs.shape) == 3, logger.error("X kp_targs shape")  # ARRAY (BATCH, NPOINTS, 3)
        assert len(kp_vis.shape) == 2, logger.error("X kp_vis shape")  # ARRAY (BATCH, NPOINTS)

        # calc euclidean distance
        diff = kp_preds - kp_targs
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=-1))

        batch_size = kp_preds.shape[0]
        for b in range(batch_size):
            for i in range(self.num_kp):
                if kp_vis[b][i]:
                    self.data[i].append(euclidean_dist[b][i])

        self.count += batch_size

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        return epe_mean

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype("float"))
        return pck

    def get_pck_all(self, threshold):
        pck_all = []
        for kp_id in range(self.num_kp):
            pck = self._get_pck(kp_id, threshold)
            if pck is None:
                continue
            pck_all.append(pck)
        if len(pck_all) == 0:
            logger.debug("No valid data in get_pck_all, you should dobule check this")
        pck_all = np.mean(np.array(pck_all))
        return pck_all

    def get_measures(self) -> Dict:
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(self.val_min, self.val_max, self.steps)
        thresholds = np.array(thresholds)
        area_under_one = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_per_kp = list()
        auc_per_kp = list()
        pck_curve_per_kp = list()

        # Create one plot for each part
        for kp_id in range(self.num_kp):
            # mean error
            mean = self._get_epe(kp_id)
            if mean is None:
                # there was no valid measurement for this keypoint
                continue
            epe_mean_per_kp.append(mean)
            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(kp_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_per_kp.append(pck_curve.copy())
            auc = np.trapz(pck_curve, thresholds)
            auc /= area_under_one  # area_under_curve / area_under_one
            auc_per_kp.append(auc)
        # Display error per keypoint
        epe_mean_all = np.mean(np.array(epe_mean_per_kp))
        auc_all = np.mean(np.array(auc_per_kp))
        # pck_curve_all = np.mean(np.array(pck_curve_per_kp), axis=0)

        return {
            "epe_mean_per_kp": np.array(epe_mean_per_kp),  # ARRAY [N_KEYPOINTS,]
            "pck_curve_per_kp": np.array(pck_curve_per_kp),  # ARRAY [N_KEYPOINTS, N_THRESHOLDS]
            "auc_per_kp": np.array(auc_per_kp),  # ARRAY [N_KEYPOINTS,]
            "epe_mean_all": epe_mean_all,
            "auc_all": auc_all,
            "thresholds": thresholds,
        }


class Joint3DPCK(_PCKMetric):
    num_kp = 21

    def __init__(self, **cfg) -> None:
        super().__init__(**cfg)
        self.eval_type = cfg.get("EVAL_TYPE", "joints_3d")
        self.reset()

    def _get_predictions(self, preds: Dict, targs: Dict):
        if self.eval_type == "joints_3d_rel":
            pred_jts_rel = preds["pred_joints_3d_rel"]
            gt_jts_rel = targs["master_joints_3d_rel"]
            return pred_jts_rel, gt_jts_rel, None

        elif self.eval_type == "joints_3d":
            pred_jts = preds["pred_joints_3d"]
            gt_jts = targs["master_joints_3d"]
            return pred_jts, gt_jts, None

        else:
            raise ValueError(f"Unknown eval_type {self.eval_type} in {type(self).__name__}")

    def __str__(self) -> str:
        return f"h3dpck: {self.get_pck_all(0.02):6.4f}"


class Vert3DPCK(_PCKMetric):
    num_kp = 778

    def __init__(self, **cfg) -> None:
        super().__init__(**cfg)
        self.eval_type = cfg.get("EVAL_TYPE", "verts_3d")
        self.reset()

    def _get_predictions(self, preds: Dict, targs: Dict):
        if self.eval_type == "verts_3d_rel":
            pred_jts_rel = preds["pred_verts_3d_rel"]
            gt_jts_rel = targs["master_verts_3d_rel"]
            return pred_jts_rel, gt_jts_rel, None

        elif self.eval_type == "verts_3d":
            pred_vts = preds["pred_verts_3d"]
            gt_vts = targs["master_verts_3d"]
            # b = pred_vts.shape[0]
            # verts_vis = torch.ones((b, 778), dtype=torch.float32)
            return pred_vts, gt_vts, None

        else:
            raise ValueError(f"Unknown eval_type {self.eval_type} in {type(self).__name__}")

    def __str__(self) -> str:
        return f"h3dpck: {self.get_pck_all(0.02):6.4f}"