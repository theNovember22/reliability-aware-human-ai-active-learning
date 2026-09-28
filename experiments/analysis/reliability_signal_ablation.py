import os
import sys
import torch

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

sys.path.insert(0, ROOT_DIR)

# ---------------------------------------------------------
# Imports from existing project
# ---------------------------------------------------------

from src.baseline.utils import (
    CustomDataset,
    Evaluator,
)

from src.uncertainty.mc_dropout import mc_dropout_predict

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

NUM_REPEATS = 5
HUMAN_REPEAT_ID = 7
LIMIT_HUMAN_PRED = 3000
MC_SAMPLES = 20

DATA_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "DR-5"
)

CHECKPOINT_DIR = os.path.join(
    ROOT_DIR,
    "checkpoints",
    "baseline",
    "dr5"
)

print("=" * 70)
print("RELIABILITY SIGNAL ABLATION")
print("=" * 70)

print("ROOT_DIR       :", ROOT_DIR)
print("DATA_DIR       :", DATA_DIR)
print("CHECKPOINT_DIR :", CHECKPOINT_DIR)
print("DEVICE         :", DEVICE)
print("MC SAMPLES     :", MC_SAMPLES)
print()