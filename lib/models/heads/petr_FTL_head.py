import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.config import CN
from ...utils.builder import HEAD
from ...utils.transform import inverse_sigmoid
from lib.models.heads.petr_head import PETRHead


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
class PETRHead_FTL(PETRHead):

    def __init__(self, cfg: CN):
        super(PETRHead_FTL, self).__init__(cfg)

        self.conv1 = nn.Sequential(nn.Conv2d(256, 3 * 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(96),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3 * 32, 3 * 32, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(96),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(3 * 32, 256, kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(256))

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def cam_P_transf(self, P, x):
        x_homo = torch.cat([x, torch.ones((*x.shape[:-1], 1), dtype=x.dtype, device=x.device)], dim=-1)
        x_homo = (P[..., :3, :] @ x_homo.transpose(2, 3)).transpose(2, 3)
        return x_homo

    def position_embeding(self, img_feat, img_metas):
        x = img_feat  # [B, N, C, H, W]
        B, N, C, H, W = x.shape
        # step1: P
        P_inv = torch.linalg.inv(img_metas["cam_intr"]) @ img_metas["cam_extr"][..., :3, :]
        P = (img_metas["cam_intr"]) @ torch.linalg.inv(img_metas["cam_extr"])[..., :3, :]
        #step2: img_ft -> p_pt conv1
        x = self.conv1(x.flatten(0, 1))  # [B*N, 3*D, H, W]
        x = x.reshape(B * N, 3, -1, H, W).permute(0, 2, 3, 4, 1)  # [B*N, D, H, W, 3]
        # 3: P * p_ptå
        x = self.cam_P_transf(P_inv, x.reshape(B, N, -1, 3))
        # 4 : p_pt -> img_ft concat 8C
        x = x.reshape(B * N, -1, H, W, 3).permute(0, 4, 1, 2, 3).reshape(B * N, -1, H, W)
        # 5: conv shape const
        x = self.conv2(x)  # [B*N, C, H, W]
        # 6: split -> B, N, C ,H, W, img->p
        x = x.reshape(B * N, 3, -1, H, W).permute(0, 2, 3, 4, 1)  # [B*N, D, H, W, 3]
        # 7: P-1 * p_pt
        x = self.cam_P_transf(P, x.reshape(B, N, -1, 3))
        # 8: p_pt -> img conv3
        x = x.reshape(B * N, -1, H, W, 3).permute(0, 4, 1, 2, 3).reshape(B, N, -1, H, W)
        x = self.conv3(x.flatten(0, 1)).reshape(B, N, -1, H, W)
        return x  #[B, N, C, H, W]

    def forward(self, mlvl_feats, img_metas, template_mesh, **kwargs):
        x = mlvl_feats[1]
        batch_size, num_cams = x.size(0), x.size(1)

        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        masks = x.new_zeros((batch_size, num_cams, inp_img_h, inp_img_w))

        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        sin_embed = self.positional_encoding(masks)
        pos_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        feature_ftl = self.position_embeding(x, img_metas)

        reference_points = self.reference_points.weight
        # query_embeds = self.query_embedding(torch.cat([pos2posemb3d(reference_points), template_mesh], dim=-1))
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()

        outs_dec, _ = self.transformer(feature_ftl, masks, query_embeds, pos_embed)
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
