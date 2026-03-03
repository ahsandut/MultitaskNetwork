import cv2
import numpy as np
from config import IMG_SIZE

SIGMA = 3

def normalize_resize(img):
    # Normalize [0,1]
    img = img.astype(np.float32)/255.0
    # Resize 320x320
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))

def generate_heatmap(coords, num_landmarks=5):
    heatmaps = np.zeros((num_landmarks, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, (x,y) in enumerate(coords):
        xx, yy = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
        heatmaps[i] = np.exp(-((xx-x)**2 + (yy-y)**2)/(2*SIGMA**2))
        heatmaps[i] /= heatmaps[i].max()
    return heatmaps