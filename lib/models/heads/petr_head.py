import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.config import CN
from ...utils.logger import logger
from ...utils.builder import HEAD
from ...utils.transform import inverse_sigmoid
from ..bricks.transformer import build_transformer
from ..layers.petr_transformer import SinePositionalEncoding3D


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature**(2 * (torch.div(dim_t, 2, rounding_mode='floor')) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


@HEAD.register_module()
class PETRHead(nn.Module):

    def __init__(self, cfg: CN):
        super(PETRHead, self).__init__()
        self.cfg_transformer = cfg.TRANSFORMER
        self.cfg_position_encoding = cfg.POSITIONAL_ENCODING

        self.num_joints = cfg.DATA_PRESET.NUM_JOINTS
        self.with_position = cfg.WITH_POSITION
        self.with_multiview = cfg.WITH_MULTIVIEW
        self.num_query = cfg.NUM_QUERY  #  should equal to the n_vertices + n_joints,
        self.depth_num = cfg.DEPTH_NUM
        self.position_dim = 3 * self.depth_num
        self.position_range = cfg.POSITION_RANGE  # [-1.2, -1.2,  0.2,  1.2, 1.2, 2.0]
        self.LID = cfg.LID
        self.depth_start = cfg.DEPTH_START  # 0.2
        self.depth_end = cfg.DEPTH_END  # 3
        self.embed_dims = cfg.EMBED_DIMS
        self.in_channels = cfg.IN_CHANNELS
        self.num_preds = cfg.NUM_PREDS
        self.num_reg_fcs = cfg.NUM_REG_FCS
        self.coord_relative = cfg.get("COORD_RELATIVE_TO_REFERENCE", False)

        self._build_head_module()
        self.init_weights()

    def _build_head_module(self):
        self.positional_encoding = SinePositionalEncoding3D(num_feats=self.cfg_position_encoding.NUM_FEATS,
                                                            normalize=self.cfg_position_encoding.NORMALIZE)
        self.transformer = build_transformer(self.cfg_transformer)
        assert self.transformer.embed_dims == self.embed_dims, "PETRHead embed_dim should be equal to PETRTransformer embed_dims"

        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 3))
        if self.coord_relative is False:
            reg_branch.append(nn.Sigmoid())
        reg_branch = nn.Sequential(*reg_branch)
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_preds)])

        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(3 + self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def position_embeding(self, img_feats, img_metas, masks=None):
        eps = 1e-5
        inp_img_h, inp_img_w = img_metas['inp_img_shape']
        B, N, C, H, W = img_feats[1].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * inp_img_h / H  # U
        coords_w = torch.arange(W, device=img_feats[0].device).float() * inp_img_w / W  # V

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_plus1 = index + 1
            bin_size = (self.depth_end - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_plus1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.depth_end - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]

        # (W, H, D, 3)
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d], indexing='ij')).permute(1, 2, 3, 0)

        # ===== 1. using camera intrinsic to convert UVD 2 XYZ  >>>>>>
        INTR = img_metas["cam_intr"]  # (B, N, 3, 3)
        fx = INTR[..., 0, 0].unsqueeze(dim=-1)  # (B, N, 1)
        fy = INTR[..., 1, 1].unsqueeze(dim=-1)  # (B, N, 1)
        cx = INTR[..., 0, 2].unsqueeze(dim=-1)  # (B, N, 1)
        cy = INTR[..., 1, 2].unsqueeze(dim=-1)  # (B, N, 1)
        cam_param = torch.cat((fx, fy, cx, cy), dim=-1)  # (B, N, 4)
        cam_param = cam_param.view(B, N, 1, 1, 1, 4).repeat(1, 1, W, H, D, 1)  # (B, N, W, H, D, 4)

        coords_uv, coords_d = coords[..., :2], coords[..., 2:3]  # (W, H, D, 2), (W, H, D, 1)
        coords_uv = coords_uv.view(1, 1, W, H, D, 2).repeat(B, N, 1, 1, 1, 1)  # (B, N, W, H, D, 2)
        coords_d = coords_d.view(1, 1, W, H, D, 1).repeat(B, N, 1, 1, 1, 1)  # (B, N, W, H, D, 1)

        coords_uv = (coords_uv - cam_param[..., 2:4]) / cam_param[..., :2]  # (B, N, W, H, D, 2)
        coords_xy = coords_uv * coords_d
        coords_z = coords_d
        coords = torch.cat((coords_xy, coords_z), dim=-1)  # (B, N, W, H, D, 3)

        # ===== 2. using camera extrinsic to transfer childs' XYZ 2 parent's space >>>>>>
        EXTR = img_metas["cam_extr"]  # (B, N, 4, 4)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords = coords.unsqueeze(-1)  # (B, N, W, H, D, 4, 1)
        EXTR = EXTR.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)  # (B, N, W, H, D, 4, 4)
        coords3d = torch.matmul(EXTR, coords).squeeze(-1)[..., :3]  # (B, N, W, H, D, 3),  xyz in parent's space

        # ===== 3. using position range to normalize coords3d
        #                 0     1     2     3     4     5
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / \
                                (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / \
                                (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / \
                                (self.position_range[5] -self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)  # (B, N, W, H, D, 3)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)  # (B, N, W, H)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)  # (B, N, H, W)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)  # (B*N, 3*D, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)  # res: (B*N, 256, H, W), 256 is self.embed_dims

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

    def forward(self, mlvl_feats, img_metas, template_mesh, **kwargs):
        x = mlvl_feats[1]
        batch_size, num_cams = x.size(0), x.size(1)

        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        masks = x.new_zeros((batch_size, num_cams, inp_img_h, inp_img_w))
        # masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
        # for img_id in range(batch_size):
        #     for cam_id in range(num_cams):
        #         masks[img_id, cam_id, :input_img_w, :input_img_h] = 0

        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
        pos_embed = coords_position_embeding

        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        pos_embed = pos_embed + sin_embed

        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(torch.cat([pos2posemb3d(reference_points), template_mesh], dim=-1))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()

        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed)
        outs_dec = torch.nan_to_num(outs_dec)

        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            if self.coord_relative is True:
                reference = inverse_sigmoid(reference_points.clone())
                assert reference.shape[-1] == 3
                tmp = self.reg_branches[lvl](outs_dec[lvl])
                tmp = tmp + reference
                outputs_coord = tmp.sigmoid()
            else:
                outputs_coord = self.reg_branches[lvl](outs_dec[lvl])
            outputs_coords.append(outputs_coord)

        all_coords_preds = torch.stack(outputs_coords)  # (N_Decoder, B, NUM_QUERY, 3)

        # scale all the joitnts and vertices to the camera space
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        all_coords_preds[..., 0:1] = \
            all_coords_preds[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        all_coords_preds[..., 1:2] = \
            all_coords_preds[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        all_coords_preds[..., 2:3] = \
            all_coords_preds[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]

        results = dict()
        results["all_coords_preds"] = all_coords_preds
        return results
