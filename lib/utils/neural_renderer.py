import torch
import neural_renderer as nr


class NeuralRenderer(torch.nn.Module):

    def __init__(
        self,
        fill_back=True,
    ):
        super(NeuralRenderer, self).__init__()
        self.fill_back = fill_back

    """ Render Depth """

    def forward(
            self,
            vertices,
            faces,
            mode,  # all
            rast_size=256,  # all
            orig_size=(256, 256),  # all
            anti_aliasing=False,  # all
            far=10.0,  # all
            R=None,  # persp 
            t=None,  # persp 
            intr=None,  # persp
            dist_coeffs=None,  # persp
            bbx=None,  # persp
            scale=None,  # ortho
            trans=None,  # ortho
    ):
        """
        mode is in ['persp','ortho']
        """
        if mode == "persp":
            # batch_size = vertices.shape(0)
            if self.fill_back:
                faces = (torch.cat(
                    (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]),
                    dim=1,
                ).to(vertices.device).detach())

            assert intr is not None, "intr must given in persp mode"

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
            vertices = self.projection(vertices, intr, R, t, dist_coeffs, orig_size, bbx=bbx)

            faces = nr.vertices_to_faces(vertices, faces)
            # rasteriation
            rast = nr.rasterize_depth(faces, rast_size, anti_aliasing, far=far)
            # normalize to 0~1
            # rend = self.normalize_depth_with_maxmin(rast, depthmax=depthmax, depthmin=depthmin)
            dep = self.normalize_depth(rast, far=far)
            mask = nr.rasterize_silhouettes(faces, rast_size, anti_aliasing)

            return dep, mask

        elif mode == "ortho":
            # batch_size = vertices.shape(0)
            if self.fill_back:
                faces = (torch.cat(
                    (faces, faces[:, :, list(reversed(range(faces.shape[-1])))]),
                    dim=1,
                ).to(vertices.device).detach())

            assert scale is not None, "scale must given in ortho mode"
            assert trans is not None, "trans must given in ortho mode"

            scale = scale.unsqueeze(1)
            trans = trans.unsqueeze(1)
            """ xyz -> uvd """
            # vertices (B,V,3)
            vertices = self.projection_ortho(vertices, scale, trans, orig_size)

            faces = nr.vertices_to_faces(vertices, faces)
            # rasteriation
            rast = nr.rasterize_depth(faces, rast_size, anti_aliasing, far=far)
            # normalize to 0~1
            # rend = self.normalize_depth_with_maxmin(rast, depthmax=depthmax, depthmin=depthmin)
            dep = self.normalize_depth(rast, far=far)
            mask = nr.rasterize_silhouettes(faces, rast_size, anti_aliasing)
            return dep, mask
        else:
            print("Mode ", mode, " is invalid")
            raise Exception()

    def normalize_depth(self, img, far):
        img_inf = torch.eq(img, far * torch.ones_like(img)).type(torch.float32)  # Bool Tensor
        img_ok = 1 - img_inf  # Bool Tensor
        img_no_back = img_ok * img  # Float Tensor
        img_max = img_no_back.max(dim=1, keepdim=True)[0]  # batch of max value
        img_max = img_max.max(dim=2, keepdim=True)[0]

        img_min = img.min(dim=1, keepdim=True)[0]  # batch of min values
        img_min = img_min.min(dim=2, keepdim=True)[0]

        new_depth = (img_max - img) / (img_max - img_min + 1e-4)
        new_depth = torch.max(new_depth, torch.zeros_like(new_depth))
        new_depth = torch.min(new_depth, torch.ones_like(new_depth))
        return new_depth

    def projection(self, vertices, K, R, t, dist_coeffs, orig_size, bbx=None, eps=1e-9):
        """
        Calculate projective transformation of vertices given a projection matrix
        Input parameters:
        K: batch_size * 3 * 3 intrinsic camera matrix
        R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
        dist_coeffs: vector of distortion coefficients
        orig_size: original size of image captured by the camera [width, height]
        Returns: For each point [X,Y,Z] in world coordinates [u,v,z]
        where u,v are the coordinates of the projection in
        pixels and z is the depth
        Modified by Li Jiasen: add bbx
        """

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
        if bbx is not None:
            u = (u - bbx[:, 0:1]) / bbx[:, 2:3] * orig_size[0]
            v = (v - bbx[:, 1:2]) / bbx[:, 3:4] * orig_size[1]

        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        v = orig_size[1] - v
        u = 2 * (u - orig_size[0] / 2.0) / orig_size[0]
        v = 2 * (v - orig_size[1] / 2.0) / orig_size[1]
        vertices = torch.stack([u, v, z], dim=-1)
        return vertices

    def projection_ortho(self, vertices, scale, trans, orig_size, eps=1e-9):
        """
        Calculate ortho transformation of vertices given a projection parameters
        Input parameters:
        s: batch_size * 1 * 2 ; scale
        t: batch_size * 1 * 2 ; trans
        orig_size: original size of image captured by the camera
        Returns: For each point [X,Y,Z] in world coordinates [u,v,z]
        where u,v are the coordinates of the projection in
        pixels and z is the depth
        """

        u, v, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
        uv = torch.stack([u, v], dim=-1)  # (B, 778, 2)
        # Notice the unit here
        uv = uv * scale + trans  # (0~256)

        u, v = uv[:, :, 0], uv[:, :, 1]
        # map u,v from [0, img_size] to [-1, 1] to use by the renderer
        u = 2 * (u - orig_size[0] / 2) / orig_size[0]
        v = 2 * (v - orig_size[1] / 2) / orig_size[1]
        v = -v

        # avoid negative value
        # z = z + 2.0
        vertices = torch.stack([u, v, z], dim=-1)
        return vertices
