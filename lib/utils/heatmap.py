import cv2
import numpy as np
import torch
import torch.nn.functional as F


def calc_dists(preds, target, normalize, mask):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))  # (njoint, B)
    for b in range(preds.size(0)):
        for j in range(preds.size(1)):
            if mask[b][j] == 0:
                dists[j, b] = -1
            elif target[b, j, 0] < 1 or target[b, j, 1] < 1:
                dists[j, b] = -1
            else:
                dists[j, b] = torch.dist(preds[b, j, :], target[b, j, :]) / normalize[b]

    return dists


def dist_acc(dist, thr=0.5):
    """ Return percentage below threshold while ignoring values with a -1 """
    dist = dist[dist != -1]
    if len(dist) > 0:
        return 1.0 * (dist < thr).sum().item() / len(dist)
    else:
        return -1


def accuracy_heatmap(output, target, mask, thr=0.5):
    """ Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First to be returned is average accuracy across 'idxs', Second is individual accuracies
    """
    preds, confi = get_heatmap_pred(output)  # (B, njoint, 2)
    gts, confi = get_heatmap_pred(target)
    norm = torch.ones(preds.size(0)) * output.size(3) / 10.0  # (B, ), all 6.4:(1/10 of heatmap side)
    dists = calc_dists(preds, gts, norm, mask)  # (njoint, B)

    acc = torch.zeros(mask.size(1))
    avg_acc = 0
    cnt = 0

    for i in range(mask.size(1)):  # njoint
        acc[i] = dist_acc(dists[i], thr)
        if acc[i] >= 0:
            avg_acc += acc[i]
            cnt += 1

    if cnt != 0:
        avg_acc /= cnt

    return avg_acc, acc


def generate_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1


def get_heatmap_pred(heatmaps):
    """get predictions from heatmaps in torch Tensor
    return type: torch.LongTensor
    """
    assert heatmaps.dim() == 4, "Score maps should be 4-dim (B, nJoints, H, W)"
    maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

    maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
    idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()  # (B, njoint, 2)

    preds[:, :, 0] = (preds[:, :, 0]) % heatmaps.size(3)  # + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / heatmaps.size(3))  # + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds, maxval.float()


def batch_with_heatmap(
    inputs,
    heatmaps,
    num_rows=2,
    parts_to_show=None,
    n_in_batch=1,
):
    batch_img = []
    for n in range(min(inputs.shape[0], n_in_batch)):
        inp = inputs[n]
        batch_img.append(sample_with_heatmap(inp, heatmaps[n], num_rows=num_rows, parts_to_show=parts_to_show))
    resu = np.concatenate(batch_img)
    return resu


def sample_with_heatmap(img, heatmap, num_rows=3, parts_to_show=None):
    if parts_to_show is None:
        parts_to_show = np.arange(heatmap.shape[0])  # 21

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows  # H // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = cv2.resize(img, (size, size))

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = cv2.resize(heatmap[part_idx], (size, size))
        out_resized = out_resized.astype(float)
        out_img = inp_small.copy() * 0.4
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * 0.6

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img


def color_heatmap(x):
    color = np.zeros((x.shape[0], x.shape[1], 3))
    color[:, :, 0] = gauss(x, 0.5, 0.6, 0.2) + gauss(x, 1, 0.8, 0.3)
    color[:, :, 1] = gauss(x, 1, 0.5, 0.3)
    color[:, :, 2] = gauss(x, 1, 0.2, 0.3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color


def gauss(x, a, b, c, d=0):
    return a * np.exp(-((x - b)**2) / (2 * c**2)) + d
