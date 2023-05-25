import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ...utils.config import CN
from ...utils.logger import logger
from ...utils.builder import HEAD
from ...utils.misc import param_size
from ...utils.transform import inverse_sigmoid, batch_cam_extr_transf, batch_cam_intr_projection
from ...utils.points_utils import sample_points_from_ball_query
from ..bricks.transformer import build_transformer
from ..layers.petr_transformer import SinePositionalEncoding3D
from pytorch3d.ops import ball_query


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


class BasePointEmbedHead(nn.Module):

    def __init__(self, cfg: CN):
        super(BasePointEmbedHead, self).__init__()
        self.cfg_transformer = cfg.TRANSFORMER
        self.cfg_position_encoding = cfg.POSITIONAL_ENCODING

        self.with_position = cfg.WITH_POSITION
        self.with_multiview = cfg.WITH_MULTIVIEW
        self.num_query = cfg.NUM_QUERY  #  should equal to the n_vertices + n_joints,
        self.depth_num = cfg.DEPTH_NUM
        self.position_dim = 3 * self.depth_num
        self.position_range = cfg.POSITION_RANGE
        self.LID = cfg.LID
        self.depth_start = cfg.DEPTH_START
        self.depth_end = cfg.DEPTH_END
        self.embed_dims = cfg.EMBED_DIMS
        self.in_channels = cfg.IN_CHANNELS
        self.num_preds = cfg.NUM_PREDS

        # custom args
        self.center_shift = cfg.get("CENTER_SHIFT", False)

        self._build_head_module()
        self.init_weights()
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters, "
                    f"and got custom args: {self._str_custom_args()}")

    def _str_custom_args(self):
        return (f"center_shift: {self.center_shift}")

    def _build_head_module(self):
        self.center_shift_layer = nn.Sequential(nn.Linear(self.num_query, self.num_query), nn.ReLU(),
                                                nn.Linear(self.num_query, 1))

        self.positional_encoding = SinePositionalEncoding3D(num_feats=self.cfg_position_encoding.NUM_FEATS,
                                                            normalize=self.cfg_position_encoding.NORMALIZE)
        self.transformer = build_transformer(self.cfg_transformer)

        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1)
        self.reg_branches = nn.ModuleList()
        for i in range(self.num_preds):
            reg_branch = nn.Sequential(nn.Linear(self.pt_feat_dim, self.pt_feat_dim), nn.ReLU(),
                                       nn.Linear(self.pt_feat_dim, 3))
            self.reg_branches.append(reg_branch)

        self.adapt_pos3d = nn.Conv2d(self.embed_dims * 3 // 2, self.embed_dims, kernel_size=1, stride=1, padding=0)
        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims * 2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 2, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.reference_embed = nn.Embedding(self.num_query, self.embed_dims)
        self.query_embedding = nn.Sequential(
            nn.Linear(6 + (self.embed_dims * 3 // 2), self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.pt_feat_dim),
        )

    def position_embeding(self, img_feat, img_metas, masks=None):
        eps = 1e-5
        inp_img_h, inp_img_w = img_metas['inp_img_shape']
        B, N, C, H, W = img_feat.shape
        coords_h = torch.arange(H, device=img_feat.device).float() * inp_img_h / H  # U
        coords_w = torch.arange(W, device=img_feat.device).float() * inp_img_w / W  # V

        if self.LID:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feat.device).float()
            index_plus1 = index + 1
            bin_size = (self.depth_end - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_plus1
        else:
            index = torch.arange(start=0, end=self.depth_num, step=1, device=img_feat.device).float()
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

        coords3d_absolute = copy.deepcopy(coords3d)
        # ===== 3. using position range to normalize coords3d
        #                 0     1     2     3     4     5
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / \
                                (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / \
                                (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / \
                                (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)  # (B, N, W, H, D, 3)

        #coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)  # (B, N, W, H)
        #coords_mask = masks | coords_mask.permute(0, 1, 3, 2)  # (B, N, H, W)
        coords3d_feat = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)  # (B*N, 3*D, H, W)
        coords3d_feat = inverse_sigmoid(coords3d_feat)
        coords_position_embeding = self.position_encoder(coords3d_feat)  # res: (B*N, 256, H, W), 256 is self.embed_dims

        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords3d, coords3d_absolute, coords_mask

    def init_weights(self):
        # The initialization for transformer is important, we leave it to the child class to decide.
        ### self.transformer.init_weights()
        nn.init.uniform_(self.reference_embed.weight.data, 0, 1)

    def forward(self, mlvl_feat, img_metas, reference_points, template_mesh, **kwargs):
        raise NotImplementedError(f"forward not implemented in base BasePointEmbedHead")


@HEAD.register_module()  # ptemb
class POEM_PositionEmbeddedAggregationHead(BasePointEmbedHead):

    def __init__(self, cfg: CN):
        self.nsample = cfg.N_SAMPLE
        self.radius = cfg.RADIUS_SAMPLE
        self.pt_feat_dim = cfg.POINTS_FEAT_DIM
        self.init_pt_feat_dim = cfg.INIT_POINTS_FEAT_DIM  # 8
        super(POEM_PositionEmbeddedAggregationHead, self).__init__(cfg)

    def _build_head_module(self):
        super(POEM_PositionEmbeddedAggregationHead, self)._build_head_module()
        self.transition_up = nn.Linear(self.init_pt_feat_dim, self.pt_feat_dim)
        logger.warning(f"{type(self).__name__} build done")

    def _str_custom_args(self):
        str_ = super(POEM_PositionEmbeddedAggregationHead, self)._str_custom_args()
        return str_ + ", " + f"init_pt_feat_dim: {self.init_pt_feat_dim}"

    def forward(self, mlvl_feat, img_metas, reference_points, template_mesh, **kwargs):
        results = dict()
        x = mlvl_feat
        batch_size, num_cams = x.size(0), x.size(1)

        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        ref_mesh_gt = img_metas["ref_mesh_gt"]
        inp_res = torch.Tensor([inp_img_w, inp_img_h]).to(x.device).float()
        masks = x.new_zeros((batch_size, num_cams, inp_img_h, inp_img_w))
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])  # [B, N, 256, H, W]
        feat_dim = x.size(2)
        assert feat_dim == self.pt_feat_dim, \
            f"self.pt_feat_dim {self.pt_feat_dim} should be equal to feat_dim {feat_dim}"
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        coords_embed, coords3d, coords3d_abs, _ = self.position_embeding(mlvl_feat, img_metas, masks)

        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())  # (B, N, 256, H, W)

        posi_embed = sin_embed + coords_embed
        x = x + posi_embed

        center_point = reference_points.mean(1).unsqueeze(1)  # [B, 1, 3]
        if self.center_shift == True:
            center_point = center_point + 0.01 * \
                           self.center_shift_layer(reference_points.transpose(1, 2)).transpose(1, 2)

        # x3d = self.input_proj_3Daware(x.flatten(0, 1))  # (BN, F*D, H, W)
        x3d = x.reshape(batch_size, num_cams, -1, self.depth_num, *x.shape[-2:])  # (B, N, F*D/D, D, H, W)
        x3d = x3d.permute(0, 1, 5, 4, 3, 2).contiguous()  # (B, N, W, H, D, F)
        assert x3d.shape[-1] == self.init_pt_feat_dim, \
            f"self.init_pt_feat_dim {self.init_pt_feat_dim} should be equal to x3d's last dim {x3d.shape[-1]}"

        reference_embed = self.reference_embed.weight  # (799, 3)
        reference_embed = pos2posemb3d(reference_embed)  # (799, 384)   384 <-- (self.embed_dims * 3 // 2)
        reference_embed = reference_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 799, 384)
        template_mesh = template_mesh.unsqueeze(0).expand(batch_size, -1, -1)  # template mesh  # (B, 799, 3)

        #*  normalize reference points into position range
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        query_embeds = self.query_embedding(torch.cat([
            reference_embed,
            reference_points,
            template_mesh,
        ], dim=-1))  # (B, 799, embed_dims=32)

        # coords3d_abs: [B, N, 32, 32, 32, 3], --> [B, M, 3]
        coords3d_abs = coords3d_abs.reshape(batch_size, -1, 3)
        x3d = x3d.reshape(batch_size, -1, self.init_pt_feat_dim)
        randlist = torch.randperm(coords3d_abs.size(1))
        coords3d_abs = coords3d_abs[:, randlist, :]
        x3d = x3d[:, randlist, :]

        #  [B, nsample, 3],  [B, nsample, 32]
        pt_xyz, x3d_nsample = sample_points_from_ball_query(pt_xyz=coords3d_abs,
                                                            pt_feats=x3d,
                                                            center_point=center_point,
                                                            k=self.nsample,
                                                            radius=self.radius)

        pt_feats = self.transition_up(x3d_nsample)
        pt_xyz[..., 0:1] = (pt_xyz[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        pt_xyz[..., 1:2] = (pt_xyz[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        pt_xyz[..., 2:3] = (pt_xyz[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        interm_ref_pts = self.transformer(
            pt_xyz=pt_xyz,  # [B, 2048, 3]
            pt_feats=pt_feats,  # [B, 2048, 256]
            query_emb=query_embeds,  # [B, 799, 256]
            query_xyz=reference_points,  # [B, 799, 3]
            reg_branches=self.reg_branches,
        )

        interm_ref_pts = torch.nan_to_num(interm_ref_pts)
        all_coords_preds = interm_ref_pts  # (N_Decoder, BS, NUM_QUERY, 3)

        # scale all the joitnts and vertices to the camera space
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        all_coords_preds[..., 0:1] = \
            all_coords_preds[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        all_coords_preds[..., 1:2] = \
            all_coords_preds[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        all_coords_preds[..., 2:3] = \
            all_coords_preds[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]

        results["all_coords_preds"] = all_coords_preds
        return results


@HEAD.register_module()  # proj_selfagg
class POEM_Projective_SelfAggregation_Head(BasePointEmbedHead):

    def __init__(self, cfg: CN):
        self.nsample = cfg.N_SAMPLE
        self.radius = cfg.RADIUS_SAMPLE
        self.pt_feat_dim = cfg.POINTS_FEAT_DIM
        self.merge_mode = cfg.get("CAM_FEAT_MERGE", "sum")
        self.query_type = cfg.get("QUERY_TYPE", "KPT")
        # POEM: I cat V_tmpl cat V_init
        # KPT: I
        # MVP: g + I
        # METRO: g cat V_tmpl
        super(POEM_Projective_SelfAggregation_Head, self).__init__(cfg)

    def _build_head_module(self):
        super(POEM_Projective_SelfAggregation_Head, self)._build_head_module()
        self.merge_net_feature = nn.ModuleList()
        self.merge_net_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                          nn.Linear(self.embed_dims, self.embed_dims // 2)))
        self.merge_net_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims // 2, self.embed_dims // 2), nn.ReLU(),
                          nn.Linear(self.embed_dims // 2, self.embed_dims)))
        self.merge_net_query_feature = nn.ModuleList()
        self.merge_net_query_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                          nn.Linear(self.embed_dims, self.embed_dims // 2)))
        self.merge_net_query_feature.append(
            nn.Sequential(nn.Linear(self.embed_dims // 2, self.embed_dims // 2), nn.ReLU(),
                          nn.Linear(self.embed_dims // 2, self.embed_dims)))

        self.layer_global_feat = nn.Linear(512, self.embed_dims)
        if self.query_type == "POEM":
            self.query_embedding = nn.Sequential(
                nn.Linear(6 + self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        elif self.query_type == "KPT":
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        elif self.query_type == "MVP":
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        elif self.query_type == "METRO":
            self.query_embedding = nn.Sequential(
                nn.Linear(self.embed_dims + 3, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.pt_feat_dim),
            )
        else:
            raise ValueError(f"no such query_type: {self.query_type}")
        logger.info(f"{type(self).__name__} got query_type: {self.query_type}")
        logger.warning(f"{type(self).__name__} build done")

    def _str_custom_args(self):
        str_ = super()._str_custom_args()
        return str_ + ", " + f"agg_merge_mode: {self.merge_mode}"

    def merge_features(self, q, merge_net, master_id, batch_size):
        """
            q: [B, nsample, N, 256]
            q_merged: [B, nsample, 256]
        """
        master_is_zero = torch.sum(master_id)
        assert master_is_zero == 0, "only support master_id is 0"
        q1 = q[:, :, 0, :]
        q = merge_net[0](q)
        master_features = q[:, :, 0, :]  # [B, nsample, 128]
        other_features = q[:, :, 1:, :]  # [B, nsample, 7, 128]
        q = torch.matmul(other_features, master_features.unsqueeze(-1))  # [B, nsample, 7, 1]
        q = torch.matmul(other_features.transpose(2, 3), q).squeeze(-1)
        q_merged = merge_net[1](q)  # [B, nsample, 256]
        q_merged = q1 + q_merged
        return q_merged

    def sample_points_from_ball_query(self, pt_xyz, center_point, k, radius):
        _, ball_idx, xyz = ball_query(center_point, pt_xyz, K=k, radius=radius, return_nn=True)
        invalid = torch.sum(ball_idx == -1) > 0
        if invalid:
            # NOTE: sanity check based on bugs reported Oct. 24
            logger.warning(f"ball query returns {torch.sum(ball_idx == -1)} / {torch.numel(ball_idx)} -1 in its index, "
                           f"which means you need to increase raidus or decrease K")
        xyz = xyz.squeeze(1)
        return xyz

    def generate_query(self, reference_embed, reference_points, template_mesh, global_feat):
        if self.query_type == "POEM":
            query_embeds = self.query_embedding(torch.cat([
                reference_embed,
                reference_points,
                template_mesh,
            ], dim=-1))  # (B, 799, pt_feat_dim)
        elif self.query_type == "KPT":
            query_embeds = self.query_embedding(reference_embed)  # (B, 799, pt_feat_dim)
        elif self.query_type == "MVP":
            query_embeds = self.query_embedding(global_feat + reference_embed)
        elif self.query_type == "METRO":
            query_embeds = self.query_embedding(torch.cat([
                global_feat,
                template_mesh,
            ], dim=-1))  # (B, 799, pt_feat_dim)
        else:
            raise ValueError(f"no such query_type: {self.query_type}")
        return query_embeds

    def forward(self, mlvl_feat, img_metas, reference_points, template_mesh, **kwargs):
        results = dict()
        x = mlvl_feat
        batch_size, num_cams = x.size(0), x.size(1)

        global_feat = kwargs.get("global_feat", None)  # (BxN, 512)
        if global_feat != None:
            global_feat = self.layer_global_feat(global_feat)  # (BxN, 256)
            global_feat = global_feat.reshape(batch_size, num_cams, -1)  # (B, N, 256)
            global_feat = global_feat.sum(1)  # (B, 256) -> (B, 799, 256)
            global_feat = global_feat.unsqueeze(1).repeat(1, self.num_query, 1)

        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        # ref_mesh_gt = img_metas["ref_mesh_gt"]
        inp_res = torch.Tensor([inp_img_w, inp_img_h]).to(x.device).float()
        masks = x.new_zeros((batch_size, num_cams, inp_img_h, inp_img_w))
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])  # [B, N, 256, H, W]
        feat_dim = x.size(2)
        assert feat_dim == self.pt_feat_dim, "self.pt_feat_dim should be equal to feat_dim"
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        coords_embed, coords3d, coords3d_abs, _ = self.position_embeding(mlvl_feat, img_metas, masks)

        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())  # (B, N, 256, H, W)

        posi_embed = sin_embed + coords_embed
        x = x + posi_embed

        center_point = reference_points.mean(1).unsqueeze(1)  # [B, 1, 3]
        if self.center_shift == True:
            center_point = center_point + 0.01 * \
                           self.center_shift_layer(reference_points.transpose(1, 2)).transpose(1, 2)

        reference_embed = self.reference_embed.weight  # (799, 256)
        # reference_embed = pos2posemb3d(reference_embed)  # (799, 384)   384 <-- (self.embed_dims * 3 // 2)
        reference_embed = reference_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 799, 256)
        template_mesh = template_mesh.unsqueeze(0).expand(batch_size, -1, -1)  # template mesh  # (B, 799, 3)

        ref_proj_2d = reference_points.unsqueeze(1).repeat(1, num_cams, 1, 1)

        #*  normalize reference points into position range
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        query_embeds = self.generate_query(
            reference_embed=reference_embed,
            reference_points=reference_points,
            template_mesh=template_mesh,
            global_feat=global_feat,
        )  # (B, 799, pt_feat_dim)

        # coords3d: [B, N, 32, 32, 32, 3], coords3d_nsmaple: [B, nsample, 3]
        coords3d_abs = coords3d_abs.reshape(batch_size, -1, 3)
        randlist = torch.randperm(coords3d_abs.size(1))
        coords3d_abs = coords3d_abs[:, randlist, :]
        coords3d_nsample = self.sample_points_from_ball_query(coords3d_abs, center_point, self.nsample, self.radius)
        pt_xyz = coords3d_nsample
        # coords3d_nsample -> [B, N, nsample, 3], coords3d_nsample_project: [B, N, nsample, 2]
        coords3d_nsample = coords3d_nsample.unsqueeze(1).repeat(1, num_cams, 1, 1)
        coords3d_nsample_project = batch_cam_extr_transf(torch.linalg.inv(img_metas["cam_extr"]), coords3d_nsample)
        coords3d_nsample_project = batch_cam_intr_projection(img_metas["cam_intr"], coords3d_nsample_project)
        # results["coords3d_nsample_project"] = coords3d_nsample_project.detach()
        coords3d_nsample_project = coords3d_nsample_project.flatten(0, 1).unsqueeze(-2)  # uv -> WH

        coords3d_nsample_project = torch.einsum("bijk, k->bijk", coords3d_nsample_project,
                                                1.0 / inp_res)  # TENSOR (B, N,  nsample, 2), [0 ~ 1]
        coords3d_nsample_project = coords3d_nsample_project * 2 - 1
        coords3d_project_invalid = (coords3d_nsample_project > 1.0) | (coords3d_nsample_project < -1.0)

        # NOTE: sanity check based on bugs reported Oct. 24
        ratio_invalid = torch.sum(coords3d_project_invalid) / torch.numel(coords3d_project_invalid)
        if ratio_invalid > 0.3 and ratio_invalid <= 0.5:
            logger.warning(f"Projection returns {torch.sum(coords3d_project_invalid)} / "
                           f"{torch.numel(coords3d_project_invalid)} outsiders in its resutls, "
                           f"may be a bug !")
        if ratio_invalid > 0.5:
            raise ValueError(f"Too many invalid projective points, "
                             f"consider redue the RADIUS_SAMPLE : {self.radius}, "
                             f"and check center_points's validness !")

        # pt_sampled_feats: [B, nsample, N, 256], sin_embed_sample: [B, nsample, N, 256]

        ref_proj_2d = batch_cam_extr_transf(torch.linalg.inv(img_metas["cam_extr"]), ref_proj_2d)
        ref_proj_2d = batch_cam_intr_projection(img_metas["cam_intr"], ref_proj_2d)
        ref_proj_2d = ref_proj_2d.flatten(0, 1).unsqueeze(-2)  # uv -> WH
        ref_proj_2d = torch.einsum("bijk, k->bijk", ref_proj_2d, 1.0 / inp_res)  # TENSOR (B, N,  nsample, 2), [0 ~ 1]
        ref_proj_2d = ref_proj_2d * 2 - 1

        query_feat = F.grid_sample(x.flatten(0, 1), ref_proj_2d, align_corners=False)\
                .squeeze(-1).reshape(batch_size, num_cams, feat_dim, self.num_query).permute(0,3,1,2)
        pt_sampled_feats = F.grid_sample(x.flatten(0, 1), coords3d_nsample_project, align_corners=False)\
                                .squeeze(-1).reshape(batch_size, num_cams, feat_dim, self.nsample).permute(0,3,1,2)
        pt_smapled_emb = F.grid_sample(posi_embed.flatten(0, 1), coords3d_nsample_project, align_corners=False)\
                                .squeeze(-1).reshape(batch_size, num_cams, feat_dim, self.nsample).permute(0,3,1,2)
        if self.merge_mode == "attn":
            master_id = img_metas["master_id"]
            pt_sampled_feats = self.merge_features(pt_sampled_feats, self.merge_net_feature, master_id, batch_size)
            query_feat = self.merge_features(query_feat, self.merge_net_query_feature, master_id, batch_size)
        elif self.merge_mode == "sum":
            pt_sampled_feats = torch.sum(pt_sampled_feats, dim=-2)  # [B, nsample, 256]
            query_feat = torch.sum(query_feat, dim=-2)
        else:
            raise ValueError(f"CAM_FEAT_MERGE must in [attn, sum], default sum, got {self.merge_mode}")

        pt_smapled_emb = torch.sum(pt_smapled_emb, dim=-2)  # [B, nsample, 256]

        pt_xyz[..., 0:1] = (pt_xyz[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        pt_xyz[..., 1:2] = (pt_xyz[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        pt_xyz[..., 2:3] = (pt_xyz[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        interm_ref_pts = self.transformer(
            pt_xyz=pt_xyz,  # [B, 2048, 3]
            pt_feats=pt_sampled_feats,  # [B, 2048, 256]
            pt_embed=pt_smapled_emb,  # [B, 2048,256]
            query_feat=query_feat,
            query_emb=query_embeds,  # [B, 799, 256]
            query_xyz=reference_points,  # [B, 799, 3]
            reg_branches=self.reg_branches,
        )
        interm_ref_pts = torch.nan_to_num(interm_ref_pts)
        all_coords_preds = interm_ref_pts  # (N_Decoder, BS, NUM_QUERY, 3)

        # scale all the joitnts and vertices to the camera space
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        all_coords_preds[..., 0:1] = \
            all_coords_preds[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        all_coords_preds[..., 1:2] = \
            all_coords_preds[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        all_coords_preds[..., 2:3] = \
            all_coords_preds[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]

        results["all_coords_preds"] = all_coords_preds
        return results


@HEAD.register_module()
# class POEM_METROlikeHead(BasePointEmbedHead):
class POEM_woPoint_Head(BasePointEmbedHead):

    def __init__(self, cfg: CN):
        self.num_reg_fcs = cfg.NUM_REG_FCS
        self.pt_feat_dim = cfg.get("POINTS_FEAT_DIM", cfg.EMBED_DIMS)
        super(POEM_woPoint_Head, self).__init__(cfg)

    def _build_head_module(self):
        super()._build_head_module()

        if hasattr(self, "reg_branches"):
            logger.info("remove BasePointEmbedHead' s reg_branch")
            del self.reg_branches

        # Reg branch with shared weights
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 3))
        # reg_branch.append(nn.Sigmoid())
        reg_branch = nn.Sequential(*reg_branch)
        self.reg_branches = nn.ModuleList([reg_branch for _ in range(self.num_preds)])
        logger.warning(f"{type(self).__name__} build done")

    def _str_custom_args(self):
        str_ = super()._str_custom_args()
        return str_ + ", " + f"num_reg_fcs: {self.num_reg_fcs}"

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_embed.weight.data, 0, 1)

    def forward(self, mlvl_feat, img_metas, reference_points, template_mesh, **kwargs):
        results = dict()
        x = mlvl_feat  # (B, N, 128, 32, 32)
        batch_size, num_cams = x.size(0), x.size(1)
        inp_img_w, inp_img_h = img_metas["inp_img_shape"]  #  (256, 256)
        masks = x.new_zeros((batch_size, num_cams, inp_img_h, inp_img_w))
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)
        coords_embed, coords3d, coords3d_abs, _ = self.position_embeding(mlvl_feat, img_metas, masks)
        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        pos_embed = coords_embed + sin_embed

        reference_embed = self.reference_embed.weight  # (799, 3)
        reference_embed = pos2posemb3d(reference_embed)  # (799, 384)   384 <-- (self.embed_dims * 3 // 2)
        reference_embed = reference_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 799, 384)
        template_mesh = template_mesh.unsqueeze(0).expand(batch_size, -1, -1)  # template mesh  # (B, 799, 3)

        #*  normalize reference points into position range
        reference_points[..., 0:1] = (reference_points[..., 0:1] - self.position_range[0]) / \
                                        (self.position_range[3] - self.position_range[0])
        reference_points[..., 1:2] = (reference_points[..., 1:2] - self.position_range[1]) / \
                                        (self.position_range[4] - self.position_range[1])
        reference_points[..., 2:3] = (reference_points[..., 2:3] - self.position_range[2]) / \
                                        (self.position_range[5] - self.position_range[2])

        query_embeds = self.query_embedding(torch.cat([
            reference_embed,
            reference_points,
            template_mesh,
        ], dim=-1))  # (B, 799, embed_dims=256)

        outs_dec, interm_ref_pts, _ = self.transformer(
            x=x,
            mask=masks,
            query_embed=query_embeds,
            pos_embed=pos_embed,
            reference_points=reference_points,
            reg_branch=self.reg_branches,
        )
        all_coords_preds = interm_ref_pts  # (N_Decoder, BS, NUM_QUERY, 3)
        # scale all the joitnts and vertices to the camera space
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        all_coords_preds[..., 0:1] = \
            all_coords_preds[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        all_coords_preds[..., 1:2] = \
            all_coords_preds[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        all_coords_preds[..., 2:3] = \
            all_coords_preds[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]

        results["all_coords_preds"] = all_coords_preds
        return results
