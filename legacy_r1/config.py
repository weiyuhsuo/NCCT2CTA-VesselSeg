import torch
from pathlib import Path

# Paths
ROOT = Path("/cpfs01/projects-SSD/cfff-71d5b2895244_SSD/hyb_24110860026/weiyushuo/NCCT2CTA-VesselSeg")
DATA_DIR = ROOT / "data" / "data_100"
PREPROC_DIR = ROOT / "data" / "preprocessed"
NCCT_DIR = DATA_DIR / "NCCT"
CTA_DIR = DATA_DIR / "CTA"
SEG_DIR = DATA_DIR / "SEG"
SRC_DIR = ROOT / "src"
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PRED_DIR = OUTPUT_DIR / "predictions"
FIG_DIR = OUTPUT_DIR / "figures"

# Data preprocessing
NCCT_CLIP = (-200, 500)
CTA_CLIP = (-200, 800)
IMAGE_SIZE = 256

# Train/Val/Test split
VAL_RATIO = 0.10
TEST_RATIO = 0.20
RANDOM_SEED = 42

# Training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_WORKERS = 0
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR = 0.5
EARLY_STOP_PATIENCE = 15

# Loss weights
LAMBDA_CTA = 0.5
LAMBDA_SEG = 0.5

# Model
IN_CHANNELS = 1
OUT_CHANNELS = 1
FEATURES = (64, 128, 256, 512)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
