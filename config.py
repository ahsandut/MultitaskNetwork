DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"

# Training hyperparameters
LR = 2e-4
WEIGHT_DECAY = 1e-5
BETAS = (0.9, 0.999)
BATCH_SIZE = 4
EPOCHS = 150
GRAD_ACCUM_STEPS = 2
PATIENCE = 20

# Input size
IMG_SIZE = 320
IN_CHANNELS = 2  # Axial + Sagittal

# Loss weights
SEG_WEIGHT = 1.2
LM_WEIGHT = 0.03

# Cross-validation
N_FOLDS = 5
NUM_LANDMARKS = 5