import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points, ball_query
from .logger import logger


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def sample_points_from_ball_query(pt_xyz, pt_feats, center_point, k, radius):
    _, ball_idx, xyz = ball_query(center_point, pt_xyz, K=k, radius=radius, return_nn=True)
    invalid = torch.sum(ball_idx == -1) > 0
    if invalid:
        logger.warning(f"ball query returns {torch.sum(ball_idx == -1)} / {torch.numel(ball_idx)} -1 in its index, "
                       f"which means you need to increase raidus or decrease K")

    points = index_points(pt_feats, ball_idx).squeeze(1)
    xyz = xyz.squeeze(1)
    return xyz, points


def sample_points_from_knn(pt_xyz, pt_feats, center_point, k):
    _, knn_idx, xyz = knn_points(center_point, pt_xyz, K=k, return_nn=True)
    points = index_points(pt_feats, knn_idx).squeeze(1)
    xyz = xyz.squeeze(1)
    return xyz, points