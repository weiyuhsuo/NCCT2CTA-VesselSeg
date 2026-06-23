"""Round 2 configuration: Dice Loss + 512x512 + segmentation-biased weights."""

import torch
from pathlib import Path

# Paths
ROOT = Path("/cpfs01/projects-SSD/cfff-71d5b2895244_SSD/hyb_24110860026/weiyushuo/NCCT2CTA-VesselSeg")
DATA_DIR = ROOT / "data" / "data_100"
PREPROC_DIR = Path("/tmp/preprocessed_512")
NCCT_DIR = DATA_DIR / "NCCT"
CTA_DIR = DATA_DIR / "CTA"
SEG_DIR = DATA_DIR / "SEG"
SRC_DIR = ROOT / "src"
OUTPUT_DIR = ROOT / "outputs_r2"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"

# Data preprocessing
NCCT_CLIP = (-200, 500)
CTA_CLIP = (-200, 800)
IMAGE_SIZE = 512  # Round 2: 512×512 for fine vessel details

# Train/Val/Test split
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_SEED = 42

# Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8  # Round 2: reduced for 512×512 (A100 80GB can handle ~8)
NUM_WORKERS = 2
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 20  # Round 2: longer patience for Dice's noisier convergence

# Loss weights — Round 2: bias toward segmentation (harder task)
LAMBDA_CTA = 0.3      # CTA generation weight
LAMBDA_SEG = 0.7      # Vessel segmentation weight
DICE_WEIGHT = 0.8     # Dice vs BCE within segmentation loss (0.8 Dice + 0.2 BCE)

# Model
IN_CHANNELS = 1
OUT_CHANNELS = 1
FEATURES = (64, 128, 256, 512)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
