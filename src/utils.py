import matplotlib.pyplot as plt
import random
from pathlib import Path

import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import (
    BATCH_SIZE,
    DATA_DIR,
    FIGURES_DIR,
    IMAGE_SIZE,
    SEED,
)

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _get_train_generator():
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return generator


def _build_transform(is_train, image_size):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation(5),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform


def _build_loader(split, is_train=False, image_size=IMAGE_SIZE):
    split_dir = DATA_DIR / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    dataset = ImageFolder(
        split_dir,
        transform=_build_transform(is_train=is_train, image_size=image_size),
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=is_train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker if is_train else None,
        generator=_get_train_generator() if is_train else None,
    )

    return loader


def train_loader_utils(image_size=IMAGE_SIZE):
    return _build_loader(split="train", is_train=True, image_size=image_size)


def val_loader_utils(image_size=IMAGE_SIZE):
    return _build_loader(split="valid", is_train=False, image_size=image_size)


def test_loader_utils(image_size=IMAGE_SIZE):
    return _build_loader(split="test", is_train=False, image_size=image_size)


def compute_per_class_metrics(confusion_matrix, class_names):
    cm = confusion_matrix.to(torch.float32)
    true_totals = cm.sum(dim=1)
    pred_totals = cm.sum(dim=0)

    per_class = []
    for idx, class_name in enumerate(class_names):
        tp = cm[idx, idx].item()
        fp = (pred_totals[idx] - cm[idx, idx]).item()
        fn = (true_totals[idx] - cm[idx, idx]).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class.append(
            {
                "class_name": class_name,
                "support": int(true_totals[idx].item()),
                "precision": round(float(precision), 6),
                "recall": round(float(recall), 6),
                "f1_score": round(float(f1_score), 6),
            }
        )

    return per_class


def save_confusion_matrix(confusion_matrix, class_names, output_path):
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = FIGURES_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix.to(torch.float32)
    row_sums = cm.sum(dim=1, keepdim=True).clamp(min=1)
    normalized_cm = (cm / row_sums).cpu().numpy()

    fig_size = max(12, int(len(class_names) * 0.35))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        normalized_cm,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Recall per class"},
    )
    plt.title("Confusion Matrix (row-normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_metric_curves(data, title_label, curve_label, y_label, show_plot=False, filename=None, output_dir=None):
    train = []
    validation = []
    for item in data:
        train_val = item[0].cpu().item() if hasattr(item[0], 'cpu') else float(item[0])
        val_val = item[1].cpu().item() if hasattr(item[1], 'cpu') else float(item[1])
        train.append(train_val)
        validation.append(val_val)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10,6))

    sns.lineplot(data=train, label="Train " + curve_label)
    sns.lineplot(data=validation, label="Validation " + curve_label)
    plt.title(title_label)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        base_output_dir = Path(output_dir) if output_dir is not None else FIGURES_DIR
        output_path = base_output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    
    if show_plot:
        plt.show(block=False)
    else:
        plt.close()

