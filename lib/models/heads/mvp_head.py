import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from manotorch.manolayer import ManoLayer
import copy

from ...utils.config import CN
from ...utils.logger import logger
from ...utils.builder import HEAD
from ...utils.misc import param_size
from ..bricks.conv import ConvBlock
from ...utils.net_utils import xavier_init
from ...utils.transform import inverse_sigmoid
from ..bricks.transformer import build_transformer
from ..layers.mvp_decoder import MvPDecoderLayer, MvPDecoder, ProjAttn


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


def get_rays_new(image_size, H, W, K, R, T, ret_rays_o=False):
    # calculate the camera origin
    ratio = W / image_size[0]
    batch = K.size(0)
    views = K.size(1)
    K = K.reshape(-1, 3, 3).float()
    R = R.reshape(-1, 3, 3).float()
    T = T.reshape(-1, 3, 1).float()
    # re-scale camera parameters
    K = copy.deepcopy(K.detach())
    K[:, :2] *= ratio
    rays_o = -torch.bmm(R.transpose(2, 1), T)
    # calculate the world coordinates of pixels
    j, i = torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    xy1 = torch.stack([i.to(K.device), j.to(K.device), torch.ones_like(i).to(K.device)], dim=-1).unsqueeze(0)
    pixel_camera = torch.bmm(
        xy1.flatten(1, 2).repeat(views * batch, 1, 1),  # *batch
        torch.inverse(K).transpose(2, 1))
    pixel_world = torch.bmm(pixel_camera - T.transpose(2, 1), R)
    rays_d = pixel_world - rays_o.transpose(2, 1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.unsqueeze(1).repeat(1, H * W, 1, 1)
    if ret_rays_o:
        return rays_d.reshape(batch, views, H, W, 3), \
               rays_o.reshape(batch, views, H, W, 3) / 1000
    else:
        return rays_d.reshape(batch, views, H, W, 3)
    """ Numpy Code
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d
    """


@HEAD.register_module()
class MVPHead(nn.Module):

    def __init__(self, cfg: CN):
        super(MVPHead, self).__init__()

        self.num_query = cfg.NUM_QUERY  #  should equal to the n_joints,
        self.depth_num = cfg.DEPTH_NUM
        self.position_dim = 3 * self.depth_num
        self.position_range = cfg.POSITION_RANGE 
        self.LID = cfg.LID
        self.depth_start = cfg.DEPTH_START  # 0.2
        self.depth_end = cfg.DEPTH_END  # 3

        self.embed_dims = cfg.EMBED_DIMS
        self.in_channels = cfg.IN_CHANNELS
        self.image_size = cfg.DECODER.IMAGE_SIZE
        self.mano_pose_ncomps = cfg.get("MANO_POSE_NCOMPS", 45)
        self.mano_shape_ncomps = 10
        self.n_joints = cfg.DATA_PRESET.NUM_JOINTS
        assert self.num_query == self.n_joints, "MVP, joints & queries mismatch"

        self.num_cams = cfg.DECODER.CAMERA_NUM

        decoder_layer = MvPDecoderLayer(cfg.POSITION_RANGE, cfg.DECODER.IMAGE_SIZE, cfg.DECODER.d_model,
                                        cfg.DECODER.dim_feedforward, cfg.DECODER.dropout, cfg.DECODER.activation,
                                        cfg.DECODER.num_feature_levels, cfg.DECODER.nhead, cfg.DECODER.dec_n_points,
                                        cfg.DECODER.detach_refpoints_cameraprj_firstlayer, cfg.DECODER.fuse_view_feats,
                                        cfg.DECODER.CAMERA_NUM, cfg.DECODER.projattn_posembed_mode,
                                        self.mano_pose_ncomps, self.mano_shape_ncomps)

        self.decoder = MvPDecoder(cfg, decoder_layer, cfg.DECODER.num_decoder_layers,
                                  cfg.DECODER.return_intermediate_dec)

        self.feat_size = cfg.FEAT_SIZE
        self.reference_feats = nn.Linear(cfg.DECODER.d_model * 3 * cfg.DECODER.CAMERA_NUM,
                                         cfg.DECODER.d_model)  # 256*feat_level*num_views

        self.num_preds = cfg.NUM_PREDS
        self.num_reg_fcs = cfg.NUM_REG_FCS

        self.input_proj = nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=1)

        self.reg_branches = nn.ModuleList()
        for i in range(self.num_preds):
            reg_branch = nn.Sequential(nn.Linear(self.embed_dims, self.embed_dims), nn.ReLU(),
                                       nn.Linear(self.embed_dims, 3))
            self.reg_branches.append(reg_branch)

        self.mano_use_pca = False if self.mano_pose_ncomps == 45 else True
        self.mano_layer = ManoLayer(joint_rot_mode="axisang",
                                    use_pca=self.mano_use_pca,
                                    ncomps=self.mano_pose_ncomps,
                                    mano_assets_root="assets/mano_v1_2",
                                    center_idx=cfg.CENTER_IDX,
                                    flat_hand_mean=True)
        self.center_idx = cfg.CENTER_IDX

        self.layer_global_feat = nn.Linear(512, self.embed_dims)
        # self.reference_points = nn.Embedding(self.num_query, 3)
        self.reference_points = nn.Linear(cfg.DECODER.d_model, 3)
        self.tgt_pose_embedding = nn.Embedding(self.num_query, 2 * self.embed_dims)
        self.query_embedding = nn.Sequential(
            nn.Linear(3 + self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, 3),
        )

        self.feat_delayer = nn.ModuleList([
            ConvBlock(self.feat_size[0], cfg.DECODER.d_model, kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[1], cfg.DECODER.d_model, kernel_size=3, relu=True, norm='bn'),
            ConvBlock(self.feat_size[2], cfg.DECODER.d_model, kernel_size=3, relu=True, norm='bn'),
        ])

        self.init_weights()
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters, "
                    f"and got custom args: mano pca: {self.mano_use_pca} | mano_pose_ncomps: {self.mano_pose_ncomps}")

    def init_weights(self):
        """Initialize weights of the transformer head."""
        for m in self.modules():
            if isinstance(m, ProjAttn):
                m._reset_parameters()
            elif isinstance(m, nn.MultiheadAttention):
                if hasattr(m, 'weight') and m.weight.dim() > 1:
                    xavier_init(m, distribution='uniform')

        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        nn.init.uniform_(self.tgt_pose_embedding.weight.data, 0, 1)

    def collate_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _get_views(self, all_feats):
        src_flatten_views = []
        mask_flatten_views = []
        spatial_shapes_views = []

        for lvl, src in enumerate(all_feats):
            BN, C, H, W = src.size()
            src = src.view(BN, C, H, W)
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_views.append(spatial_shape)
            mask = src.new_zeros(bs, h, w).bool()
            mask_flatten_views.append(mask)
            mask = mask.flatten(1)
            src_flatten_views.append(src)

        spatial_shapes_views = torch.as_tensor(spatial_shapes_views, dtype=torch.long, device=mask.device)
        level_start_index_views = torch.cat((mask.new_zeros(
            (1,), dtype=torch.long), torch.as_tensor(spatial_shapes_views, dtype=torch.long,
                                                     device=mask.device).prod(1).cumsum(0)[:-1]))
        valid_ratios_views = torch.stack([self.get_valid_ratio(m) for m in mask_flatten_views], 1)
        mask_flatten_views = [m.flatten(1) for m in mask_flatten_views]
        return src_flatten_views, mask_flatten_views, spatial_shapes_views, level_start_index_views, valid_ratios_views

    def _get_ray_for_mvp(self, img_metas, batch, all_feats):

        N = img_metas["cam_intr"].shape[1]  #[B, N, 3, 3]

        cam_R = img_metas["cam_extr"][:, :, :3, :3]
        cam_T = img_metas["cam_extr"][:, :, :-1, -1:]
        cam_K_crop = img_metas["cam_intr"]
        nfeat_level = len(all_feats)
        camera_rays = []
        # get pos embed, camera ray or 2d coords
        for lvl in range(nfeat_level):
            # this can be compute only once, without iterating over views
            camera_rays.append(
                get_rays_new(self.image_size, all_feats[lvl].shape[-2], all_feats[lvl].shape[-1], cam_K_crop, cam_R,
                             cam_T).flatten(0, 1))

        return camera_rays

    def forward(self, mlvl_feats, img_metas, template_mesh, **kwargs):
        """img_feats for ResNet 34: 
                torch.Size([BN, 64, 64, 64])
                torch.Size([BN, 128, 32, 32])
                torch.Size([BN, 256, 16, 16])
                torch.Size([BN, 512, 8, 8])
        """
        batch_size = mlvl_feats[1].size(0)
        all_feats = mlvl_feats[::-1]
        all_feats = all_feats[:3]

        all_feats[0] = self.feat_delayer[0](all_feats[0].flatten(0, 1))
        all_feats[1] = self.feat_delayer[1](all_feats[1].flatten(0, 1))
        all_feats[2] = self.feat_delayer[2](all_feats[2].flatten(0, 1))

        feats_0 = F.adaptive_avg_pool2d(all_feats[0], (1, 1))
        feats_1 = F.adaptive_avg_pool2d(all_feats[1], (1, 1))
        feats_2 = F.adaptive_avg_pool2d(all_feats[2], (1, 1))
        feats = torch.cat((feats_0, feats_1, feats_2), dim=1).squeeze().view(batch_size, -1)
        ref_feats = self.reference_feats(feats).unsqueeze(1)

        tgt_pose = self.tgt_pose_embedding.weight.unsqueeze(0).repeat(batch_size, 1, 1).sigmoid()  #[B, 21, 512]
        tgt, query_embed = torch.split(tgt_pose, self.embed_dims, dim=-1)  #[B, 21, 256]
        reference_points = self.reference_points(query_embed + ref_feats).sigmoid()


        camera_rays = self._get_ray_for_mvp(img_metas, batch_size, all_feats)
        (src_flatten_views, mask_flatten_views, spatial_shapes_views, level_start_index_views, valid_ratios_views) = \
                                                                                            self._get_views(all_feats)

        _, inter_ref_points, inter_mano_params = self.decoder(tgt,
                                                              reference_points,
                                                              src_flatten_views,
                                                              camera_rays,
                                                              meta=img_metas,
                                                              src_spatial_shapes=spatial_shapes_views,
                                                              src_level_start_index=level_start_index_views,
                                                              src_valid_ratios=valid_ratios_views,
                                                              query_pos=query_embed,
                                                              reg_branches=self.reg_branches,
                                                              src_padding_mask=mask_flatten_views)

        # inter_references: [6, B, 21, 3], mano_theta_beta: [6, B, mano_ncomps]
        inter_ref_points = torch.nan_to_num(inter_ref_points)
        inter_mano_params = torch.nan_to_num(inter_mano_params)

        outputs_coords_verts = []

        for lvl in range(inter_mano_params.shape[0]):
            mano_pose = inter_mano_params[lvl, :, :(3 + self.mano_pose_ncomps)]
            mano_shape = inter_mano_params[lvl, :, (3 + self.mano_pose_ncomps):]
            assert mano_shape.size(-1) == self.mano_shape_ncomps, "inter_mano_params shape mismatch"
            mano_out = self.mano_layer(mano_pose, mano_shape)
            outputs_coords_v = mano_out.verts
            joints_root = inter_ref_points[lvl, :, self.center_idx, :].unsqueeze(1)  # [B, 1, 3]
            outputs_coords_v += joints_root
            outputs_coords_verts.append(outputs_coords_v)

        all_coords_preds_joints = inter_ref_points  # (N_Decoder, B, 21, 3)
        all_coords_preds_verts = torch.stack(outputs_coords_verts)  # (N_Decoder, B, 778, 3)
        all_coords_preds = torch.cat((all_coords_preds_joints, all_coords_preds_verts),
                                     dim=-2)  # (N_Decoder, B, 799, 3)

        # scale all the joitnts and vertices to the camera space
        # position_range [xmin, ymin, zmin, xmax, ymax, zmax]
        all_coords_preds[..., 0:1] = \
            all_coords_preds[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        all_coords_preds[..., 1:2] = \
            all_coords_preds[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        all_coords_preds[..., 2:3] = \
            all_coords_preds[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]

        results = dict()
        results["all_coords_preds"] = all_coords_preds  # [6, B, 799, 3]
        results["mano_pose_shape"] = inter_mano_params  # [6, B, 58]
        return results
