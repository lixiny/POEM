

import torch
from manotorch.manolayer import ManoLayer

from .pytorch3d_renderer import Pytorch3dRenderer


class MultiviewSilhouetteLoss(torch.nn.Module):

    def __init__(
        self,
        device: torch.device,
        mano_layer: ManoLayer,
        img_size=256,
        silh_size=128,
        # vis=False,
    ):
        super(MultiviewSilhouetteLoss, self).__init__()
        self.device = device
        self.hand_face = mano_layer.th_faces.detach().clone().to(self.device)
        self.img_size = img_size
        self.silh_size = silh_size
        self.renderer = Pytorch3dRenderer(
            device=self.device,
            image_size=self.silh_size,
        )

    def forward(self, cam_intr, cam_extr, hand_verts, mask):
        """
        Args:
            cam_intr: camera intrinsics [NPERSP, 3, 3]
            cam_extr: camera extrinsics [NPERSP, 4, 4]
            hand_verts: [778, 3]
            mask: green screen mask [NPERSP, silh_size, silh_size]
        """

        # * get 2d hand verts
        n_persp = cam_intr.shape[0]
        cam_hand_verts = hand_verts[None].repeat(n_persp, 1, 1)  # (NPERSP, 778, 3)
        cam_hand_verts = torch.einsum("bij,bkj->bik", cam_hand_verts,
                                      cam_extr[:, :3, :3]) + cam_extr[:, :3, [3]].transpose(-1, -2)

        # * >>> differentiable rendering
        pred_mask = self.renderer(
            # mode="persp",
            vertices=cam_hand_verts,
            faces=self.hand_face[None].expand(n_persp, -1, -1),
            intr=cam_intr,
            orig_size=self.img_size,
        )
        # * <<< differentiable rendering

        # diff = torch.abs(pred_mask - crop_mask) * pred_mask
        diff = torch.abs(pred_mask - mask)

        return diff.mean()
