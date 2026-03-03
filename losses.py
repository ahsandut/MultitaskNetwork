import torch
import torch.nn.functional as F
import config


def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def focal_loss(pred, target):
    pred = torch.sigmoid(pred)

    pos_mask = (target >= config.THRESHOLD).float()
    neg_mask = (target < config.THRESHOLD).float()

    pos_loss = -pos_mask * (1 - pred)**config.FOCAL_GAMMA * torch.log(pred + 1e-6)
    neg_loss = -neg_mask * (pred)**config.FOCAL_GAMMA * torch.log(1 - pred + 1e-6)

    return (pos_loss + neg_loss).mean()


def multitask_loss(seg_pred, seg_gt, lm_pred, lm_gt):
    seg_loss = dice_loss(seg_pred, seg_gt)
    lm_loss = focal_loss(lm_pred, lm_gt)
    total = config.ALPHA * seg_loss + config.BETA * lm_loss
    return total, seg_loss, lm_loss