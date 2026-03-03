# MultiTask KneeNet

## Overview
This repository implements a **multi-task deep learning network** for knee MRI analysis. The network jointly performs:

1. **Segmentation** of the patellar ligament  
2. **Landmark detection** of key patellofemoral structures:
   - Trochlear groove  
   - Medial & lateral patellar facets  
   - Medial & lateral posterior condyles  

The network uses a **shared encoder** and two task-specific decoders. Joint supervision enhances cross-task learning, improves anatomical consistency, and reduces overfitting.

---

## Architecture

- **Input:** 1×320×320 grayscale MRI slice (axial or sagittal)  
- **Encoder:** 4 down-sampling stages with double 3×3 convolutions, BN + ReLU, channels: 32 → 64 → 128 → 256  
- **Segmentation Decoder:** skip connections, 1×1 conv output → binary mask  
- **Landmark Decoder:** skip connections, 1×1 conv output → Gaussian heatmaps (5 landmarks)  

**Loss function:**  
\[
L_{total} = 1.2 \times L_{dice} + 0.03 \times L_{focal}
\]

**Training:**  
- Adam optimizer, lr=2e-4, weight decay=1e-5  
- Gradient accumulation: 2 steps  
- ReduceLROnPlateau scheduler  
- Early stopping (patience=20)  
- 5-fold subject-wise cross-validation  

**Metrics:** DSC, IoU, ASD, HD, RMSE

---


## Placeholder weights

- `checkpoints/best_fold0.pth` … `best_fold4.pth` are **randomly initialized weights**.
- These allow reviewers to **load the model and run inference**, but metrics will not match the paper.
- Full trained weights are **not shared** due to patient data privacy.

