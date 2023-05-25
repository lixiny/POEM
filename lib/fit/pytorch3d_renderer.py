import numpy as np
import torch
import torch.nn as nn

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
)
from pytorch3d.renderer.cameras import OrthographicCameras


class Pytorch3dRenderer(nn.Module):

    def __init__(
        self,
        device,
        image_size=256,
        # dup_face_back=True,
    ):
        super().__init__()
        # self.dup_face_back = dup_face_back
        # pytorch 3d
        self.sigma = 1e-6
        self.cameras = OrthographicCameras(focal_length=1.0, principal_point=((0, 0),))
        self.raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1.0 / 1e-4 - 1.0) * self.sigma,
            # blur_radius=0,
            faces_per_pixel=100,
        )
        self.shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=self.sigma))
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings,
            ),
            shader=self.shader,
        ).to(device)

    def forward(
        self,
        vertices,
        faces,
        intr,
        R=None,
        t=None,
        dist_coeffs=None,
        orig_size=256,
    ):
        # if self.dup_face_back:
        #     faces = (
        #         torch.cat(
        #             (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]),
        #             dim=1,
        #         )
        #         .to(vertices.device)
        #         .detach()
        #     )

        if R is None:
            R = (torch.Tensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]).view((1, 3, 3)).to(vertices.device))

        if t is None:
            t = torch.Tensor([0, 0, 0]).view((1, 3)).to(vertices.device)

        if dist_coeffs is None:
            dist_coeffs = torch.Tensor([[0.0, 0.0, 0.0, 0.0, 0.0]]).to(vertices.device)
        """ xyz -> uvd """
        vertices = self.projection(vertices, intr, R, t, dist_coeffs, orig_size)

        # create pytorch3d meshes
        # vertices_rgb = torch.ones_like(vertices)  # (1, V, 3)
        # textures = TexturesVertex(verts_features=vertices_rgb.to(vertices.device))
        meshes_to_render = Meshes(verts=vertices, faces=faces)

        # rasterization
        image = self.renderer(meshes_to_render)
        alpha = image[..., 3]
        return alpha

    def projection(
        self,
        vertices,
        K,
        R,
        t,
        dist_coeffs,
        orig_size,
        eps=1e-9,
    ):
        # instead of P*x we compute x'*P'
        vertices = torch.matmul(vertices, R.transpose(2, 1)) + t
        x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        x_ = x / (z + eps)
        y_ = y / (z + eps)

        # Get distortion coefficients from vector
        k1 = dist_coeffs[:, None, 0]
        k2 = dist_coeffs[:, None, 1]
        p1 = dist_coeffs[:, None, 2]
        p2 = dist_coeffs[:, None, 3]
        k3 = dist_coeffs[:, None, 4]

        # we use x_ for x' and x__ for x'' etc.
        r = torch.sqrt(x_**2 + y_**2)
        x__ = x_ * (1 + k1 * (r**2) + k2 * (r**4) + k3 * (r**6)) + 2 * p1 * x_ * y_ + p2 * (r**2 + 2 * x_**2)
        y__ = y_ * (1 + k1 * (r**2) + k2 * (r**4) + k3 * (r**6)) + p1 * (r**2 + 2 * y_**2) + 2 * p2 * x_ * y_
        vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
        vertices = torch.matmul(vertices, K.transpose(1, 2))
        u, v = vertices[:, :, 0], vertices[:, :, 1]

        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        v = orig_size - v
        u = 2 * (u - orig_size / 2.0) / orig_size
        v = 2 * (v - orig_size / 2.0) / orig_size
        # convert ndc
        u = -u
        vertices = torch.stack([u, v, z], dim=-1)
        return vertices
