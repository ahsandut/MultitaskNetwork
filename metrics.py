import numpy as np
import torch
from scipy.spatial.distance import cdist


def dice_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).cpu().numpy()
    target = target.cpu().numpy()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 2 * intersection / (union + 1e-6)


def iou_score(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).cpu().numpy()
    target = target.cpu().numpy()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return intersection / (union + 1e-6)


def surface_points(mask):
    from skimage import measure
    contours = measure.find_contours(mask, 0.5)
    if len(contours)==0:
        return np.zeros((1,2))
    return np.vstack(contours)


def asd_hd(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).cpu().numpy()[0,0]
    target = target.cpu().numpy()[0,0]

    ps = surface_points(pred)
    gs = surface_points(target)

    dists = cdist(ps, gs)
    asd = dists.min(axis=1).mean()
    hd = dists.min(axis=1).max()

    return asd, hd


def heatmap_to_coords(hm):
    hm = torch.sigmoid(hm)
    B,C,H,W = hm.shape
    coords=[]
    for c in range(C):
        idx = torch.argmax(hm[0,c])
        y = idx // W
        x = idx % W
        coords.append((x.item(), y.item()))
    return np.array(coords)


def rmse(pred, gt):
    p = heatmap_to_coords(pred)
    g = heatmap_to_coords(gt)
    return np.sqrt(((p-g)**2).mean())