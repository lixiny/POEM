"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

import code
import logging

import numpy as np
import scipy
import torch
from torch import nn
from transformers.models.bert.modeling_bert import (BertConfig, BertEmbeddings, BertEncoder, BertPooler,
                                                    BertPreTrainedModel)


class BertLayerNorm(nn.Module):

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class METRO_Encoder(BertPreTrainedModel):

    def __init__(self, config):
        super(METRO_Encoder, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.img_dim = config.img_feature_dim

        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if self.use_img_layernorm:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        # self.apply(self.init_weights)
        self.init_weights()

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self,
                img_feats,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None):

        batch_size = len(img_feats)
        seq_length = len(img_feats[0])
        input_ids = torch.zeros([batch_size, seq_length], dtype=torch.long).cuda()

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        position_embeddings = self.position_embeddings(position_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                            )  # We can specify head_mask for each layer
            # head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        # Project input token features to have spcified hidden size
        img_embedding_output = self.img_embedding(img_feats)

        # We empirically observe that adding an additional learnable position embedding leads to more stable training
        embeddings = position_embeddings + img_embedding_output

        if self.use_img_layernorm:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        encoder_outputs = self.encoder(embeddings, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        outputs = (sequence_output,)
        if self.config.output_hidden_states:
            all_hidden_states = encoder_outputs[1]
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            all_attentions = encoder_outputs[-1]
            outputs = outputs + (all_attentions,)

        return outputs


class METROBlock(BertPreTrainedModel):
    """
    The archtecture of a transformer encoder block we used in METRO
    """

    def __init__(self, config):
        super(METROBlock, self).__init__(config)
        self.config = config
        self.bert = METRO_Encoder(config)
        self.cls_head = nn.Linear(config.hidden_size, self.config.output_feature_dim)
        self.residual = nn.Linear(config.img_feature_dim, self.config.output_feature_dim)
        # self.apply(self.init_weights)
        self.init_weights()

    def forward(
        self,
        img_feats,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        masked_lm_labels=None,
        next_sentence_label=None,
        position_ids=None,
        head_mask=None,
    ):
        """
        # self.bert has three outputs
        # predictions[0]: output tokens
        # predictions[1]: all_hidden_states, if enable "self.config.output_hidden_states"
        # predictions[2]: attentions, if enable "self.config.output_attentions"
        """
        predictions = self.bert(
            img_feats=img_feats,
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        # We use "self.cls_head" to perform dimensionality reduction. We don't use it for classification.
        pred_score = self.cls_head(predictions[0])
        res_img_feats = self.residual(img_feats)
        pred_score = pred_score + res_img_feats

        if self.config.output_attentions and self.config.output_hidden_states:
            return pred_score, predictions[1], predictions[-1]
        else:
            return pred_score


class METRO_Hand_Network(torch.nn.Module):
    """
    End-to-end METRO network for hand pose and mesh reconstruction from a single image.
    """

    def __init__(self, config, backbone, trans_encoder):
        super(METRO_Hand_Network, self).__init__()
        self.config = config
        self.backbone = backbone
        self.trans_encoder = trans_encoder
        self.upsampling = torch.nn.Linear(195, 778)
        self.cam_param_fc = torch.nn.Linear(3, 1)
        self.cam_param_fc2 = torch.nn.Linear(195 + 21, 150)
        self.cam_param_fc3 = torch.nn.Linear(150, 3)

    def forward(self, images, mesh_model, mesh_sampler, meta_masks=None, is_train=False):
        batch_size = images.size(0)
        # Generate T-pose template mesh
        template_pose = torch.zeros((1, 48)).to(images.device)
        template_betas = torch.zeros((1, 10)).to(images.device)
        # template_vertices, template_3d_joints = mesh_model.layer(template_pose, template_betas)
        # template_vertices = template_vertices / 1000.0
        # template_3d_joints = template_3d_joints / 1000.0
        mano_out = mesh_model(template_pose, template_betas)
        template_vertices = mano_out.verts
        template_3d_joints = mano_out.joints

        template_vertices_sub = mesh_sampler.downsample(template_vertices)

        # normalize
        template_root = template_3d_joints[:, mesh_model.center_idx, :]
        template_3d_joints = template_3d_joints - template_root[:, None, :]
        template_vertices = template_vertices - template_root[:, None, :]
        template_vertices_sub = template_vertices_sub - template_root[:, None, :]
        num_joints = template_3d_joints.shape[1]

        # concatinate template joints and template vertices, and then duplicate to batch size
        ref_vertices = torch.cat([template_3d_joints, template_vertices_sub], dim=1)
        ref_vertices = ref_vertices.expand(batch_size, -1, -1)

        # extract global image feature using a CNN backbone
        image_feat = self.backbone(images)

        # concatinate image feat and template mesh
        image_feat = image_feat.view(batch_size, 1, 2048).expand(-1, ref_vertices.shape[-2], -1)
        features = torch.cat([ref_vertices, image_feat], dim=2)

        if is_train == True:
            # apply mask vertex/joint modeling
            # meta_masks is a tensor of all the masks, randomly generated in dataloader
            # we pre-define a [MASK] token, which is a floating-value vector with 0.01s
            constant_tensor = torch.ones_like(features).cuda() * 0.01
            features = features * meta_masks + constant_tensor * (1 - meta_masks)
        # forward pass
        if self.config.output_attentions == True:
            features, hidden_states, att = self.trans_encoder(features)
        else:
            features = self.trans_encoder(features)

        pred_3d_joints = features[:, :num_joints, :]
        pred_vertices_sub = features[:, num_joints:, :]

        # learn camera parameters
        x = self.cam_param_fc(features)
        x = x.transpose(1, 2)
        x = self.cam_param_fc2(x)
        x = self.cam_param_fc3(x)
        cam_param = x.transpose(1, 2)
        cam_param = cam_param.squeeze()

        temp_transpose = pred_vertices_sub.transpose(1, 2)
        pred_vertices = self.upsampling(temp_transpose)
        pred_vertices = pred_vertices.transpose(1, 2)

        if self.config.output_attentions == True:
            return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att
        else:
            return cam_param, pred_3d_joints, pred_vertices_sub, pred_vertices


class SparseMM(torch.autograd.Function):
    """Redefine sparse @ dense matrix multiplication to enable backpropagation.
    The builtin matrix multiplication operation does not support backpropagation in some cases.
    """

    @staticmethod
    def forward(ctx, sparse, dense):
        ctx.req_grad = dense.requires_grad
        ctx.save_for_backward(sparse)
        return torch.matmul(sparse, dense)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        sparse, = ctx.saved_tensors
        if ctx.req_grad:
            grad_input = torch.matmul(sparse.t(), grad_output)
        return None, grad_input


def spmm(sparse, dense):
    return SparseMM.apply(sparse, dense)


def scipy_to_pytorch(A, U, D):
    """Convert scipy sparse matrices to pytorch sparse matrix."""
    ptU = []
    ptD = []

    for i in range(len(U)):
        u = scipy.sparse.coo_matrix(U[i])
        i = torch.LongTensor(np.array([u.row, u.col]))
        v = torch.FloatTensor(u.data)
        ptU.append(torch.sparse.FloatTensor(i, v, u.shape))

    for i in range(len(D)):
        d = scipy.sparse.coo_matrix(D[i])
        i = torch.LongTensor(np.array([d.row, d.col]))
        v = torch.FloatTensor(d.data)
        ptD.append(torch.sparse.FloatTensor(i, v, d.shape))

    return ptU, ptD


def adjmat_sparse(adjmat, nsize=1):
    """Create row-normalized sparse graph adjacency matrix."""
    adjmat = scipy.sparse.csr_matrix(adjmat)
    if nsize > 1:
        orig_adjmat = adjmat.copy()
        for _ in range(1, nsize):
            adjmat = adjmat * orig_adjmat
    adjmat.data = np.ones_like(adjmat.data)
    for i in range(adjmat.shape[0]):
        adjmat[i, i] = 1
    num_neighbors = np.array(1 / adjmat.sum(axis=-1))
    adjmat = adjmat.multiply(num_neighbors)
    adjmat = scipy.sparse.coo_matrix(adjmat)
    row = adjmat.row
    col = adjmat.col
    data = adjmat.data
    i = torch.LongTensor(np.array([row, col]))
    v = torch.from_numpy(data).float()
    adjmat = torch.sparse.FloatTensor(i, v, adjmat.shape)
    return adjmat


def get_graph_params(filename, nsize=1):
    """Load and process graph adjacency matrix and upsampling/downsampling matrices."""
    data = np.load(filename, encoding='latin1', allow_pickle=True)
    A = data['A']
    U = data['U']
    D = data['D']
    U, D = scipy_to_pytorch(A, U, D)
    A = [adjmat_sparse(a, nsize=nsize) for a in A]
    return A, U, D


class MeshSampler(object):
    """Mesh object that is used for handling certain graph operations."""

    def __init__(self,
                 filename="assets/mano_downsampling.npz",
                 num_downsampling=1,
                 nsize=1,
                 device=torch.device('cuda')):
        self._A, self._U, self._D = get_graph_params(filename=filename, nsize=nsize)
        # self._A = [a.to(device) for a in self._A]
        self._U = [u.to(device) for u in self._U]
        self._D = [d.to(device) for d in self._D]
        self.num_downsampling = num_downsampling

    def downsample(self, x, n1=0, n2=None):
        """Downsample mesh."""
        if self._D[0].device != x.device:
            print("WARNING: MeshSampler.downsample: device mismatch")
            self._D = [d.to(x.device) for d in self._D]
        if n2 is None:
            n2 = self.num_downsampling
        if x.ndimension() < 3:
            for i in range(n1, n2):
                x = spmm(self._D[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in range(n1, n2):
                    y = spmm(self._D[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x

    def upsample(self, x, n1=1, n2=0):
        """Upsample mesh."""
        if self._U[0].device != x.device:
            print("WARNING: MeshSampler.upsample: device mismatch")
            self._U = [u.to(x.device) for u in self._U]
        if x.ndimension() < 3:
            for i in reversed(range(n2, n1)):
                x = spmm(self._U[i], x)
        elif x.ndimension() == 3:
            out = []
            for i in range(x.shape[0]):
                y = x[i]
                for j in reversed(range(n2, n1)):
                    y = spmm(self._U[j], y)
                out.append(y)
            x = torch.stack(out, dim=0)
        return x
