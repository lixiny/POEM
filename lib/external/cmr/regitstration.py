import traceback

import cv2
import numpy as np
from lib.utils.logger import logger
from scipy.optimize import minimize


def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area


def distance(x, y):
    return np.sqrt(((x - y)**2).sum())


def find_1Dproj(points):
    angles = [(0, 90), (-15, 75), (-30, 60), (-45, 45), (-60, 30), (-75, 15)]
    axs = [(np.array([[np.cos(x / 180 * np.pi),
                       np.sin(x / 180 * np.pi)]]), np.array([np.cos(y / 180 * np.pi),
                                                             np.sin(y / 180 * np.pi)])) for x, y in angles]
    proj = []
    for ax in axs:
        x = (points * ax[0]).sum(axis=1)
        y = (points * ax[1]).sum(axis=1)
        proj.append([x.min(), x.max(), y.min(), y.max()])

    return np.array(proj)


def align_poly(t, poly, vertex, K, size):
    proj = np.matmul(K, (vertex + t).T).T
    proj = (proj / proj[:, 2:])[:, :2]
    proj = find_1Dproj(proj) / size
    loss = (proj - poly)**2

    return loss.mean()


def align_uv(t, uv, vertex2xyz, K):
    xyz = vertex2xyz + t
    proj = np.matmul(K, xyz.T).T
    uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
    loss = (proj - uvz)**2

    return loss.mean()


def registration_one(vertex, vertex2xyz, uv, K, size, uv_conf=None, poly=None):
    """
    Adaptive 2D-1D registration
    :param vertex: 3D mesh xyz
    :param uv: 2D pose
    :param j_regressor: matrix for vertex -> joint
    :param K: camera parameters
    :param size: image size
    :param uv_conf: 2D pose confidence
    :param poly: contours from silhouette
    :return: camera-space vertex
    """
    t = np.array([0, 0, 0.6])
    bounds = ((None, None), (None, None), (0.3, 2))
    poly_protect = [0.06, 0.02]

    try_poly = True
    if uv_conf is None:
        uv_conf = np.ones([uv.shape[0], 1])
    uv_select = uv_conf > 0.1
    if uv_select.sum() == 0:
        success = False
    else:
        loss = np.array([5])
        attempt = 5
        while loss.mean() > 2 and attempt:
            attempt -= 1
            uv = uv[uv_select.repeat(2, axis=1)].reshape(-1, 2)
            uv_conf = uv_conf[uv_select].reshape(-1, 1)
            vertex2xyz = vertex2xyz[uv_select.repeat(3, axis=1)].reshape(-1, 3)
            try:
                sol = minimize(align_uv, t, method='SLSQP', bounds=bounds, args=(uv, vertex2xyz, K))
            except Exception as e:
                logger.error(traceback.format_exc())
                success = False
                break

            t = sol.x
            success = sol.success
            xyz = vertex2xyz + t
            proj = np.matmul(K, xyz.T).T
            uvz = np.concatenate((uv, np.ones([uv.shape[0], 1])), axis=1) * xyz[:, 2:]
            loss = abs((proj - uvz).sum(axis=1))
            uv_select = loss < loss.mean() + loss.std()
            if uv_select.sum() < 13:
                break
            uv_select = uv_select[:, np.newaxis]

    if poly is not None and try_poly:
        poly = find_1Dproj(poly[0]) / size
        sol = minimize(align_poly, np.array([0, 0, 0.6]), method='SLSQP', bounds=bounds, args=(poly, vertex, K, size))
        if sol.success:
            success = sol.success
            t2 = sol.x
            d = distance(t, t2)
            if d > poly_protect[0]:
                t = t2
            elif d > poly_protect[1]:
                t = t * (1 - (d - poly_protect[1]) /
                         (poly_protect[0] - poly_protect[1])) + t2 * ((d - poly_protect[1]) /
                                                                      (poly_protect[0] - poly_protect[1]))

    return vertex + t, t, success
