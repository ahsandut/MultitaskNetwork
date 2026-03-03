import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from model import MultiTaskKneeNet
from dataset import KneeDataset
from losses import multitask_loss
from metrics import dice_score, iou_score, asd_hd, rmse
import config
import numpy as np

def train_fold(train_dataset, val_dataset, fold):
    model = MultiTaskKneeNet().to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                 betas=config.BETAS, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    best_dice = 0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        optimizer.zero_grad()
        for i, (img, seg, lm) in enumerate(train_loader):
            img, seg, lm = img.to(config.DEVICE), seg.to(config.DEVICE), lm.to(config.DEVICE)
            seg_out, lm_out = model(img)
            loss, _, _ = multitask_loss(seg_out, seg, lm_out, lm)
            loss = loss / config.GRAD_ACCUM_STEPS
            loss.backward()

            if (i+1) % config.GRAD_ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Validation
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=1)
        dices = []
        ious = []
        asds = []
        hds = []
        rmses = []

        with torch.no_grad():
            for img, seg, lm in val_loader:
                img, seg, lm = img.to(config.DEVICE), seg.to(config.DEVICE), lm.to(config.DEVICE)
                seg_out, lm_out = model(img)

                dices.append(dice_score(seg_out, seg))
                ious.append(iou_score(seg_out, seg))
                asd_val, hd_val = asd_hd(seg_out, seg)
                asds.append(asd_val)
                hds.append(hd_val)
                rmses.append(rmse(lm_out, lm))

        mean_dice = np.mean(dices)
        print(f"Fold {fold} Epoch {epoch} - Dice: {mean_dice:.4f}")

        scheduler.step(1-mean_dice)

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), f"best_fold{fold}.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    return best_dice, model