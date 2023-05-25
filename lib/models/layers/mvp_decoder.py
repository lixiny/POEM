# Copyright 2021 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ------------------------------------------------------------------------
# Multi-view Pose transformer
# ----------------------------------------------------------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# and Deformable Detr
# (https://github.com/fundamentalvision/Deformable-DETR)
# ----------------------------------------------------------------------------------------------------------------------------------------

import copy
import cv2
import warnings
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
from torch.autograd import Function
from torch.autograd.function import once_differentiable

try: 
    import Deformable as DF
except ImportError:
    warnings.warn("Deformable not found. Please install Deformable from thirdparty/Deformable")

from lib.utils.transform import inverse_sigmoid, batch_cam_extr_transf, batch_cam_intr_projection


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input " "for _is_power_of_2: " "{} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class DeformFunction(Function):

    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights,
                im2col_step):
        ctx.im2col_step = im2col_step
        output = DF.deform_forward(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                                   attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, value_level_start_index, sampling_locations,
         attention_weights) = ctx.saved_tensors

        grad_value, grad_sampling_loc, grad_attn_weight = \
            DF.deform_backward(
                value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights,
                grad_output, ctx.im2col_step)

        return \
            grad_value, \
            None, None, \
            grad_sampling_loc, \
            grad_attn_weight, None


class ProjAttn(nn.Module):

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, projattn_posembed_mode='use_rayconv'):
        """
        Projective Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling
        points per attention head per feature level

        :param projattn_posembed_mode      the
        positional embedding mode of projective attention
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, ' 'but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of
        # 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in Deform to "
                          "make the dimension of each attention "
                          "head a power of 2 which is more efficient "
                          "in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        if projattn_posembed_mode == 'use_rayconv':
            self.rayconv = nn.Linear(d_model + 3, d_model)
        elif projattn_posembed_mode == 'use_2d_coordconv':
            self.rayconv = nn.Linear(d_model + 2, d_model)
        elif projattn_posembed_mode == 'ablation_not_use_rayconv':
            self.rayconv = nn.Linear(d_model, d_model)
        else:
            raise ValueError("invalid projective attention posembed mode")
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self.projattn_posembed_mode = projattn_posembed_mode

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) \
                 * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])\
            .view(self.n_heads, 1, 1, 2).\
            repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.rayconv.weight.data)
        constant_(self.rayconv.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self,
                query,
                reference_points,
                src_views,
                camera_ray_embeds,
                input_spatial_shapes,
                input_level_start_index,
                input_padding_mask=None):
        """
        :param query                       (n_views, Length_{query}, C)
        :param reference_points
        (n_views, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (n_views, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :src_views                         list of (n_views, C, H, W), size n_levels, [(n_views, C, H_0, W_0), (n_views, C, H_0, W_0), ..., (n_views, C, H_{L-1}, W_{L-1})]
        :param input_flatten               (n_views, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (n_views, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (n_views, Length_{query}, C)
        """

        n_views, Len_q, c = query.shape
        feat_lvls = len(src_views)
        sample_grid = torch.clamp(reference_points * 2.0 - 1.0, -1.1, 1.1)
        ref_point_feat_views_alllvs = []

        if 'use_rayconv' in self.projattn_posembed_mode or \
                'use_2d_coordconv' in self.projattn_posembed_mode:
            for lvl in range(feat_lvls):
                # src_views[lvl] = src_views[lvl].repeat(1, int(self.d_model / src_views[lvl].shape[1]), 1, 1)
                ref_point_feat_views_alllvs.append(
                    F.grid_sample(src_views[lvl], sample_grid[:, :, lvl:lvl + 1, :],
                                  align_corners=False).squeeze(-1).permute(0, 2, 1))
            input_flatten = torch.cat([torch.cat([src.flatten(2) for src in src_views], dim=-1).permute(0, 2, 1), \
                                        torch.cat([cam.flatten(1,2) for cam in camera_ray_embeds], dim=1)], dim=-1)

        elif self.projattn_posembed_mode == 'ablation_not_use_rayconv':
            for lvl in range(feat_lvls):
                ref_point_feat_views_alllvs.append(
                    F.grid_sample(src_views[lvl], sample_grid[:, :, lvl:lvl + 1, :],
                                  align_corners=False).squeeze(-1).permute(0, 2, 1))
            input_flatten = torch.cat([src.flatten(2) for src in src_views], dim=-1).permute(0, 2, 1)

        n_views, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        value = self.rayconv(input_flatten)

        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(n_views, Len_in, self.n_heads, self.d_model // self.n_heads)

        ## combine the view-specific ref point feature and the joint-specific query feature
        sampling_offsets = self.sampling_offsets(torch.stack(ref_point_feat_views_alllvs, dim=2) +
                                                 query.unsqueeze(2)).view(n_views, Len_q, self.n_heads, feat_lvls,
                                                                          self.n_points, 2)
        attention_weights = self.attention_weights(
            torch.stack(ref_point_feat_views_alllvs, dim=2) + query.unsqueeze(2)).view(
                n_views, Len_q, self.n_heads, feat_lvls * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(n_views, Len_q, self.n_heads, feat_lvls,
                                                                  self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(
                reference_points.shape[-1]))
        output = DeformFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations,
                                      attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


class MvPDecoderLayer(nn.Module):

    def __init__(self,
                 position_range,
                 img_size,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation="relu",
                 n_levels=4,
                 n_heads=8,
                 n_points=4,
                 detach_refpoints_cameraprj=True,
                 fuse_view_feats='mean',
                 n_views=8,
                 projattn_posembed_mode='use_rayconv',
                 mano_pose_ncomps=15,
                 mano_shape_ncomps=10):
        super().__init__()

        # projective attention
        self.proj_attn = ProjAttn(d_model, n_levels, n_heads, n_points, projattn_posembed_mode)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # ffn mano
        self.mano_ncomps = 3 + mano_pose_ncomps + mano_shape_ncomps
        self.linear_mano_1 = nn.Linear(21 * d_model, d_model)
        self.dropout5 = nn.Dropout(dropout)
        self.linear_mano_2 = nn.Linear(d_model, self.mano_ncomps)
        self.dropout6 = nn.Dropout(dropout)
        self.norm4 = nn.LayerNorm(self.mano_ncomps)

        self.img_size = img_size

        self.detach_refpoints_cameraprj = detach_refpoints_cameraprj
        self.n_views = n_views
        self.fuse_view_feats = fuse_view_feats
        if self.fuse_view_feats == 'cat_proj':
            self.fuse_view_projction = nn.Linear(d_model * n_views, d_model)
        elif self.fuse_view_feats == 'cat_catcoord_proj':
            self.fuse_view_projction = nn.Linear((d_model + 2) * n_views, d_model)
        elif self.fuse_view_feats == 'cat_catcoord_catref_proj':
            self.fuse_view_projction = nn.Linear((d_model + 2) * n_views + 3, d_model)
        elif self.fuse_view_feats == 'sum_proj':
            self.fuse_view_projction = nn.Linear(d_model, d_model)
        elif self.fuse_view_feats == 'attn_fuse_subtract':
            self.attn_proj = nn.Sequential(*[nn.ReLU(), nn.Linear(d_model, d_model)])
        elif self.fuse_view_feats == 'cat_attn_proj':
            raise NotImplementedError
        elif self.fuse_view_feats == 'attn_fuse_dot_prod_proj':
            self.fuse_view_projction = nn.Linear(d_model, d_model)
        elif self.fuse_view_feats == 'attn_fuse_subtract_proj':
            self.attn_proj = nn.Sequential(*[nn.ReLU(), nn.Linear(d_model, d_model)])
            self.fuse_view_projction = nn.Linear(d_model, d_model)

        self.position_range = position_range
        # self.ffn_mano = nn.Sequential(nn.Linear(21 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, 58))

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_mano(self, tgt):
        tgt = self.linear_mano_2(self.dropout5(self.activation(self.linear_mano_1(tgt))))
        tgt = self.norm4(tgt)
        return tgt

    def norm2absolute(self, norm_coords):
        norm_coords[
            ...,
            0:1] = norm_coords[..., 0:1] * (self.position_range[3] - self.position_range[0]) + self.position_range[0]
        norm_coords[
            ...,
            1:2] = norm_coords[..., 1:2] * (self.position_range[4] - self.position_range[1]) + self.position_range[1]
        norm_coords[
            ...,
            2:3] = norm_coords[..., 2:3] * (self.position_range[5] - self.position_range[2]) + self.position_range[2]
        return norm_coords

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src_views,
                src_views_with_rayembed,
                src_spatial_shapes,
                level_start_index,
                meta,
                src_padding_mask=None):

        batch_size = query_pos.shape[0]
        device = query_pos.device
        nviews = self.n_views
        nfeat_level = len(src_views)
        nbins = reference_points.shape[1]

        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt_expand = tgt.unsqueeze(1).expand((-1, nviews, -1, -1)).flatten(0, 1)
        query_pos_expand = query_pos.unsqueeze(1).expand((-1, nviews, -1, -1)).flatten(0, 1)
        src_padding_mask_expand = torch.cat(src_padding_mask, dim=1)

        ref_points_proj2d_xy_norm = []

        if self.detach_refpoints_cameraprj:
            reference_points = reference_points.detach()

        reference_points_expand = reference_points.unsqueeze(1).expand(-1, nviews, -1, -1, -1)
        reference_points_expand_flatten = reference_points_expand.contiguous().view(batch_size, nviews, nbins, 3)

        reference_points_absolute = self.norm2absolute(reference_points_expand_flatten)  # [B, N, 21, 3]
        reference_points_projected2d_xy = batch_cam_extr_transf(torch.linalg.inv(meta["cam_extr"]),
                                                                reference_points_absolute)
        reference_points_projected2d_xy = batch_cam_intr_projection(meta["cam_intr"], reference_points_projected2d_xy)

        ref_points_expand = reference_points_projected2d_xy.flatten(0, 1).unsqueeze(2)
        ref_points_expand = ref_points_expand.expand(-1, -1, nfeat_level, -1) \
                            * src_spatial_shapes.flip(-1).float() \
                            / (src_spatial_shapes.flip(-1)-1).float()
        ref_points_expand = ref_points_expand / torch.max(ref_points_expand)
        tgt2 = self.proj_attn(self.with_pos_embed(tgt_expand, query_pos_expand), ref_points_expand, src_views,
                              src_views_with_rayembed, src_spatial_shapes, level_start_index, src_padding_mask_expand)

        tgt2 = tgt2.view(batch_size, nviews, nbins, -1)  # [B*N, 21, 256] -> [B, N, 21, 256]

        # various ways to fuse the multi-view feats
        if self.fuse_view_feats == 'mean':
            tgt2 = tgt2.mean(1)
        elif self.fuse_view_feats == 'cat_proj':
            tgt2 = tgt2.permute(0, 2, 1, 3).contiguous().view(batch_size, nbins, -1)  #[B, 21, N*256]
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'cat_catcoord_proj':
            tgt2 = torch.cat([tgt2, torch.stack(ref_points_proj2d_xy_norm).squeeze(-2)], dim=-1)
            tgt2 = tgt2.permute(0, 2, 1, 3)\
                .contiguous()\
                .view(batch_size, nbins, -1)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'cat_catcoord_catref_proj':
            tgt2 = \
                torch.cat(
                    [
                        tgt2,
                        torch.stack(ref_points_proj2d_xy_norm).squeeze(-2)],
                    dim=-1)
            tgt2 = tgt2.permute(0, 2, 1, 3)\
                .contiguous().view(batch_size, nbins, -1)
            tgt2 = torch.cat([tgt2, reference_points.squeeze(-2)], dim=-1)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'sum_proj':
            tgt2 = self.fuse_view_projction(tgt2.sum(1))
        elif self.fuse_view_feats == 'attn_fuse_dot_prod':
            attn_weight = \
                torch.matmul(
                    tgt2.permute(0, 2, 1, 3),
                    tgt.unsqueeze(-1)).softmax(-2)
            tgt2 = (tgt2.transpose(1, 2) * attn_weight).sum(-2)
        elif self.fuse_view_feats == 'attn_fuse_subtract':
            attn_weight = self.attn_proj(tgt2 - tgt.unsqueeze(1))
            tgt2 = (attn_weight * tgt2).sum(1)
        elif self.fuse_view_feats == 'attn_fuse_dot_prod_proj':
            attn_weight = \
                torch.matmul(
                    tgt2.permute(0, 2, 1, 3),
                    tgt.unsqueeze(-1)).softmax(-2)
            tgt2 = (tgt2.transpose(1, 2) * attn_weight).sum(-2)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'attn_fuse_subtract_proj':
            attn_weight = self.attn_proj(tgt2 - tgt.unsqueeze(1))
            tgt2 = (attn_weight * tgt2).sum(1)
            tgt2 = self.fuse_view_projction(tgt2)
        elif self.fuse_view_feats == 'cat_attn_proj':
            raise NotImplementedError
        else:
            raise NotImplementedError

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)  # [B, 21, 256]

        # ffn
        tgt = self.forward_ffn(tgt)
        tgt_mano = self.forward_mano(tgt.flatten(1, 2))
        return tgt, tgt_mano  # [B, 21, 256], [B, 25]


class MvPDecoder(nn.Module):

    def __init__(self, cfg, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        reference_points,
        src_views,
        src_views_with_rayembed,
        meta,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        reg_branches,
        query_pos=None,
        src_padding_mask=None,
    ):

        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_mano = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            output, mano_params = layer(output, query_pos, reference_points_input, src_views, src_views_with_rayembed,
                                        src_spatial_shapes, src_level_start_index, meta, src_padding_mask)

            # hack implementation for iterative pose refinement
            tmp = reg_branches[lid](output)
            new_reference_points = tmp + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_mano.append(mano_params)
                intermediate_reference_points.append(new_reference_points)

            reference_points = new_reference_points  #.detach()

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(intermediate_mano)
        else:
            return output, reference_points, mano_params
