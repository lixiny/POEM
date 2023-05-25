import numpy as np
import torch


def batch_triangulate_dlt_torch(kp2ds, Ks, Extrs):
    """torch: Triangulate multiple 2D points from multiple sets of multiviews using the DLT algorithm.
    NOTE: Expend to Batch and nJoints dimension.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD, 
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (B, N, J, 2).
        Ks (torch.Tensor): Shape: (B, N, 3, 3).
        Extrs (torch.Tensor): Shape: (B, N, 4, 4).

    Returns:
        torch.Tensor: Shape: (B, J, 3).
    """
    # assert kp2ds.shape[0] == Ks.shape[0] == Extrs.shape[0], "batch shape mismatch"
    # assert kp2ds.shape[1] == Ks.shape[1] == Extrs.shape[1], "nCams shape mismatch"
    # assert kp2ds.shape[-1] == 2, "keypoints must be 2D"
    # assert Ks.shape[-2:] == (3, 3), "K must be 3x3"
    # assert Extrs.shape[-2:] == (4, 4), "Extr must be 4x4"

    nJoints = kp2ds.shape[-2]
    batch_size = kp2ds.shape[0]
    nCams = kp2ds.shape[1]

    Pmat = Extrs[..., :3, :]  # (B, N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (B, N, 3, 4)
    Mmat = Mmat.unsqueeze(1).repeat(1, nJoints, 1, 1, 1)  # (B, J, N, 3, 4)
    Mmat = Mmat.reshape(batch_size * nJoints, nCams, *Pmat.shape[-2:])  # (BxJ, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (BxJ, N, 1, 4)

    # kp2ds: (B, N, J, 2) -> (B, J, N, 2) -> (BxJ, N, 2) -> (BxJ, N, 2, 1)
    kp2ds = kp2ds.permute(0, 2, 1, 3).reshape(batch_size * nJoints, nCams, 2).unsqueeze(3)  # (BxJ, N, 2, 1)
    A = kp2ds * M_row2  # (BxJ, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (BxJ, N, 2, 4)
    A = A.reshape(batch_size * nJoints, -1, 4)  # (BxJ, 2xN, 4)

    U, D, VT = torch.linalg.svd(A)  # VT: (BxJ, 4, 4)
    X = VT[:, -1, :3] / (VT[:, -1, 3:] + 1e-7)  # (BxJ, 3) # normalize
    X = X.reshape(batch_size, nJoints, 3)  # (B, J, 3)
    return X


def triangulate_dlt_torch(kp2ds, Ks, Extrs):
    """torch: Triangulate multiple 2D points from one set of multiviews using the DLT algorithm.
    NOTE: Expend to nJoints dimension.

    see Hartley & Zisserman section 12.2 (p.312) for info on SVD, 
    see Hartley & Zisserman (2003) p. 593 (see also p. 587).

    Args:
        kp2ds (torch.Tensor): Shape: (N, J, 2).
        Ks (torch.Tensor): Shape: (N, 3, 3).
        Extrs (torch.Tensor): Shape: (N, 4, 4).

    Returns:
        torch.Tensor: Shape: (J, 3).
    """
    nJoints = kp2ds.shape[-2]

    Pmat = Extrs[:, :3, :]  # (N, 3, 4)
    Mmat = torch.matmul(Ks, Pmat)  # (N, 3, 4)
    Mmat = Mmat.unsqueeze(0).repeat(nJoints, 1, 1, 1)  # (J, N, 3, 4)
    M_row2 = Mmat[..., 2:3, :]  # (J, N, 1, 4)

    kp2ds = kp2ds.permute(1, 0, 2).unsqueeze(3)  # (J, N, 2, 1)
    A = kp2ds * M_row2  # (J, N, 2, 4)
    A = A - Mmat[..., :2, :]  # (J, N, 2, 4)
    A = A.reshape(nJoints, -1, 4)  # (J, 2xN, 4)

    U, D, VT = torch.linalg.svd(A)  # VT: (J, 4, 4)
    X = VT[:, -1, :3] / VT[:, -1, 3:]  # (J, 3) # normalize
    return X


def triangulate_one_point_dlt(points_2d_set, Ks, Extrs):
    """Triangulate one point from one set of multiviews using the DLT algorithm.
    Implements a linear triangulation method to find a 3D
    point. For example, see Hartley & Zisserman section 12.2 (p.312).
    for info on SVD, see Hartley & Zisserman (2003) p. 593 (see also p. 587)

    Args:
        points_2d_set (set): first element is the camera index, second element is the 2d point, shape: (2,)
        Ks (np.ndarray): Camera intrinsics. Shape: (N, 3, 3). 
        Extrs (np.ndarray): Camera extrinsics. Shape: (N, 4, 4).

    Returns:
        np.ndarray: Triangulated 3D point. Shape: (3,).
    """
    A = []
    for n, pt2d in points_2d_set:
        K = Ks[int(n)]  #  (3, 3)
        Extr = Extrs[int(n)]  # (4, 4)
        P = Extr[:3, :]  # (3, 4)
        M = K @ P  # (3, 4)
        row_2 = M[2, :]
        x, y = pt2d[0], pt2d[1]
        A.append(x * row_2 - M[0, :])
        A.append(y * row_2 - M[1, :])
    # Calculate best point
    A = np.array(A)
    u, d, vt = np.linalg.svd(A)
    X = vt[-1, 0:3] / vt[-1, 3]  # normalize
    return X


def triangulate_dlt(pts, confis, Ks, Extrs, confi_thres=0.5):
    """Triangulate multiple 2D points from one set of multiviews using the DLT algorithm.
    Args:
        pts (np.ndarray): 2D points in the image plane. Shape: (N, J, 2).
        confis (np.ndarray): Confidence scores of the points. Shape: (N, J,).
        Ks (np.ndarray): Camera intrinsics. Shape: (N, 3, 3).
        Extrs (np.ndarray): Camera extrinsics. Shape: (N, 4, 4).
        confi_thres (float): Threshold of confidence score.
    Returns:
        np.ndarray: Triangulated 3D points. Shape: (N, J, 3).
    """

    assert pts.ndim == 3 and pts.shape[-1] == 2
    assert confis.ndim == 2 and confis.shape[0] == pts.shape[0]
    assert Ks.ndim == 3 and Ks.shape[1:] == (3, 3)
    assert Extrs.ndim == 3 and Extrs.shape[1:] == (4, 4)
    assert Ks.shape[0] == Extrs.shape[0] == pts.shape[0]

    dtype = pts.dtype
    nJoints = pts.shape[1]
    p3D = np.zeros((nJoints, 3), dtype=dtype)

    for j, conf in enumerate(confis.T):
        while True:
            sel_cam_idx = np.where(conf > confi_thres)[0]
            if confi_thres <= 0:
                break
            if len(sel_cam_idx) <= 1:
                confi_thres -= 0.05
                # print('confi threshold too high, decrease to', confi_thres)
            else:
                break
        points_2d_set = []
        for n in sel_cam_idx:
            points_2d = pts[n, j, :]
            points_2d_set.append((str(n), points_2d))
        p3D[j, :] = triangulate_one_point_dlt(points_2d_set, Ks, Extrs)
    return p3D


def test_batch_triangulate_dlt_torch():
    from scripts.viz_multiview_dataset import DEXYCB_3D_CONFIG
    from lib.utils.config import CN
    from lib.datasets.dexycb import DexYCBMultiView

    cfg = CN(DEXYCB_3D_CONFIG)
    cfg.MASTER_SYSTEM = "as_constant_camera"
    dataset = DexYCBMultiView(cfg)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    for i, sample in enumerate(dataloader):
        joints_2d = sample['target_joints_2d']  # (B, N, J, 2)
        Ks = sample['target_cam_intr']  # (B, N, 3, 3)
        Extrs = torch.linalg.inv(sample['target_cam_extr'])  # (B, N, 4, 4)
        P3d = batch_triangulate_dlt_torch(joints_2d, Ks, Extrs)
        master_joints_3d = sample["master_joints_3d"]  # (B, J, 3)

        diff = P3d - master_joints_3d
        diff_norm = torch.norm(diff, dim=-1)  # meter
        print(diff_norm)
        # assert False


def test_triangulate_dlt():
    from scripts.viz_multiview_dataset import DEXYCB_3D_CONFIG
    from lib.utils.config import CN
    from lib.datasets.dexycb import DexYCBMultiView

    cfg = CN(DEXYCB_3D_CONFIG)
    cfg.MASTER_SYSTEM = "as_constant_camera"
    dataset = DexYCBMultiView(cfg)

    for i in range(len(dataset)):
        sample = dataset[i]
        joints_2d = sample['target_joints_2d']  # (N, J, 2)
        print(joints_2d.shape)
        Ks = sample['target_cam_intr']  # (N, 3, 3)
        Extrs = np.linalg.inv(sample['target_cam_extr'])  # (N, 4, 4)
        confis = np.ones((joints_2d.shape[0], joints_2d.shape[1]))  # (N, J)

        P3d = triangulate_dlt(joints_2d, confis, Ks, Extrs)
        master_joints_3d = sample["master_joints_3d"]
        print(P3d - master_joints_3d)
        # assert False


if __name__ == "__main__":
    test_batch_triangulate_dlt_torch()