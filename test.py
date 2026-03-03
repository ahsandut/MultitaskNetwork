import torch
from torch.utils.data import DataLoader
from dataset import KneeDataset
from model import MultiTaskKneeNet
from metrics import dice_score, iou_score, asd_hd, rmse
import config
import numpy as np

def test_model(model_path, test_dataset):
    model = MultiTaskKneeNet().to(config.DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    loader = DataLoader(test_dataset, batch_size=1)

    dices, ious, asds, hds, rmses = [], [], [], [], []

    with torch.no_grad():
        for img, seg, lm in loader:
            img, seg, lm = img.to(config.DEVICE), seg.to(config.DEVICE), lm.to(config.DEVICE)
            seg_out, lm_out = model(img)

            dices.append(dice_score(seg_out, seg))
            ious.append(iou_score(seg_out, seg))
            asd_val, hd_val = asd_hd(seg_out, seg)
            asds.append(asd_val)
            hds.append(hd_val)
            rmses.append(rmse(lm_out, lm))

    print(f"Test DSC: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"Test IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Test ASD: {np.mean(asds):.4f} ± {np.std(asds):.4f}")
    print(f"Test HD: {np.mean(hds):.4f} ± {np.std(hds):.4f}")
    print(f"Test RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")