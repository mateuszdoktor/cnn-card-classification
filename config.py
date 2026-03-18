from pathlib import Path

import torch

# Device configuration
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

SEED = 42
BATCH_SIZE = 128
EPOCH_NUM = 20
SHOW_PLOTS = False
ARCH_NAME = "mobilenet"
USE_PRETRAINED_WEIGHTS = True
IMAGE_SIZE = 224

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VALID_DIR = DATA_DIR / "valid"
TEST_DIR = DATA_DIR / "test"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def get_run_name(arch_name=ARCH_NAME, use_pretrained_weights=USE_PRETRAINED_WEIGHTS):
    init_tag = "pretrained" if use_pretrained_weights else "scratch"
    return f"{arch_name}_{init_tag}"


def get_run_paths(arch_name=ARCH_NAME, use_pretrained_weights=USE_PRETRAINED_WEIGHTS):
    run_name = get_run_name(arch_name=arch_name, use_pretrained_weights=use_pretrained_weights)
    run_reports_dir = REPORTS_DIR / run_name
    run_figures_dir = FIGURES_DIR / run_name
    run_checkpoints_dir = CHECKPOINTS_DIR / run_name

    return {
        "run_name": run_name,
        "reports_dir": run_reports_dir,
        "figures_dir": run_figures_dir,
        "checkpoints_dir": run_checkpoints_dir,
        "metrics_path": run_reports_dir / "metrics.json",
        "per_class_metrics_path": run_reports_dir / "per_class_metrics.json",
        "confusion_matrix_path": run_figures_dir / "confusion_matrix.png",
        "best_checkpoint_path": run_checkpoints_dir / "best_model.pt",
    }

