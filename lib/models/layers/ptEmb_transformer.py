import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np

from ...utils.builder import TRANSFORMER
from ...utils.net_utils import xavier_init
from ...utils.transform import inverse_sigmoid
from ...utils.logger import logger
from ...utils.misc import param_size
from ...utils.points_utils import index_points
from ..bricks.point_transformers import (
    ptTransformerBlock,
    ptTransformerBlock_CrossAttn,
)

@TRANSFORMER.register_module()
class PtEmbedTRv2(nn.Module):

    def __init__(self, cfg):
        super(PtEmbedTRv2, self).__init__()
        self._is_init = False

        self.nblocks = cfg.N_BLOCKS
        self.nneighbor = cfg.N_NEIGHBOR
        self.nneighbor_query = cfg.N_NEIGHBOR_QUERY
        self.nneighbor_decay = cfg.get("N_NEIGHBOR_DECAY", True)
        self.transformer_dim = cfg.TRANSFORMER_DIM
        self.feat_dim = cfg.POINTS_FEAT_DIM
        self.with_point_embed = cfg.WITH_POSI_EMBED

        self.predict_inv_sigmoid = cfg.get("PREDICT_INV_SIGMOID", False)

        self.feats_self_attn = ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor)
        self.query_feats_cross_attn = nn.ModuleList()
        self.query_self_attn = nn.ModuleList()

        for i in range(self.nblocks):
            self.query_self_attn.append(ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor_query))
            self.query_feats_cross_attn.append(
                ptTransformerBlock_CrossAttn(self.feat_dim,
                                             self.transformer_dim,
                                             self.nneighbor,
                                             expand_query_dim=False))

        # self.init_weights()
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def init_weights(self):
        if self._is_init == True:
            return

        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        logger.info(f"{type(self).__name__} init done")

    def forward(self, pt_xyz, pt_feats, query_xyz, reg_branches, query_feat=None, pt_embed=None, query_emb=None):
        if pt_embed is not None and self.with_point_embed:
            pt_feats = pt_feats + pt_embed

        if query_feat is None:
            query_feats = query_emb
        else:
            query_feats = query_feat + query_emb

        pt_feats, _ = self.feats_self_attn(pt_xyz, pt_feats)

        query_xyz_n = []
        query_feats_n = []

        # query_feats = query_emb
        for i in range(self.nblocks):
            query_feats, _ = self.query_self_attn[i](query_xyz, query_feats)

            query = torch.cat((query_xyz, query_feats), dim=-1)

            query_feats, _ = self.query_feats_cross_attn[i](pt_xyz, pt_feats, query)

            if self.predict_inv_sigmoid:
                query_xyz = reg_branches[i](query_feats) + inverse_sigmoid(query_xyz)
                query_xyz = query_xyz.sigmoid()
            else:
                query_xyz = reg_branches[i](query_feats) + query_xyz

            query_xyz_n.append(query_xyz)
            query_feats_n.append(query_feats)

        return torch.stack(query_xyz_n)

