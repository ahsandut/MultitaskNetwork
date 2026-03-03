import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from config import IMG_SIZE

class KneeDataset(Dataset):
    def __init__(self, image_paths, seg_paths, heatmap_paths, augment=True):
        self.image_paths = image_paths
        self.seg_paths = seg_paths
        self.heatmap_paths = heatmap_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def random_augment(self, img, seg, heatmap):
        # Scale
        scale = random.uniform(0.8, 1.2)
        img = np.clip(img * scale, 0, 1)

        # Horizontal flip
        if random.random() > 0.5:
            img = np.flip(img, axis=1)
            seg = np.flip(seg, axis=1)
            heatmap = np.flip(heatmap, axis=2)

        # Rotation
        angle = random.uniform(-5, 5)
        M = cv2.getRotationMatrix2D((IMG_SIZE/2, IMG_SIZE/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)
        seg = cv2.warpAffine(seg, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_NEAREST)
        for i in range(heatmap.shape[0]):
            heatmap[i] = cv2.warpAffine(heatmap[i], M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR)

        # Gaussian noise
        img += np.random.normal(0, 0.05, img.shape)
        img = np.clip(img, 0, 1)

        # Gaussian blur
        img = cv2.GaussianBlur(img, (3,3), 0)
        return img, seg, heatmap

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx]).astype(np.float32) # shape: (2,H,W)
        seg = np.load(self.seg_paths[idx])[np.newaxis,:,:].astype(np.float32)
        lm = np.load(self.heatmap_paths[idx]).astype(np.float32)

        if self.augment:
            img, seg, lm = self.random_augment(img, seg, lm)
        return torch.from_numpy(img), torch.from_numpy(seg), torch.from_numpy(lm)