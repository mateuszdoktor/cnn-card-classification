import torch
import torch.nn as nn
import json
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall

from src.classifier import CardClassifier
from src.utils import (
    compute_per_class_metrics,
    save_confusion_matrix,
    set_seed,
    test_loader_utils,
)
from config import (
    DEVICE,
    ARCH_NAME,
    SEED,
    IMAGE_SIZE,
    USE_PRETRAINED_WEIGHTS,
    get_run_paths,
)


def evaluate_model():
    set_seed(SEED)
    run_paths = get_run_paths(
        arch_name=ARCH_NAME,
        use_pretrained_weights=USE_PRETRAINED_WEIGHTS,
    )

    device = torch.device(DEVICE)
    print(f"Using: {DEVICE}")
    print(f"Architecture: {ARCH_NAME}")
    print(f"Run: {run_paths['run_name']}")

    loader = test_loader_utils(image_size=IMAGE_SIZE)

    checkpoint_path = run_paths["best_checkpoint_path"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Best checkpoint not found. "
            f"Expected: {checkpoint_path}. "
            "Run training first for this ARCH_NAME/USE_PRETRAINED_WEIGHTS configuration."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_arch = checkpoint.get("arch_name")
    if checkpoint_arch is not None and checkpoint_arch != ARCH_NAME:
        raise ValueError(
            f"Checkpoint architecture mismatch: checkpoint={checkpoint_arch}, config={ARCH_NAME}"
        )

    class_names = loader.dataset.classes
    num_classes = int(checkpoint["num_classes"])
    if len(class_names) != num_classes:
        raise ValueError(
            "Dataset class count does not match checkpoint class count: "
            f"dataset={len(class_names)}, checkpoint={num_classes}"
        )

    checkpoint_class_names = checkpoint.get("class_names")
    if checkpoint_class_names is not None and checkpoint_class_names != class_names:
        raise ValueError("Dataset class-name ordering does not match checkpoint metadata.")

    random_baseline = 1.0 / num_classes

    print(f"Classes: {num_classes}")
    print(f"Random baseline accuracy: {random_baseline:.4f}")

    net = CardClassifier(
        arch_name=ARCH_NAME,
        num_classes=num_classes,
        pretrained=False,
    )
    net = net.to(device)
    net.load_state_dict(checkpoint["model_state_dict"])

    print(
        "Loaded checkpoint: "
        f"epoch={checkpoint.get('epoch')}, "
        f"best_val_accuracy={checkpoint.get('best_val_accuracy', 0.0):.4f}, "
        f"path={checkpoint_path}"
    )

    criterion = nn.CrossEntropyLoss().to(device)
    metric_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    metric_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    metric_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    metric_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    metric_confusion = ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    net.eval()
    with torch.no_grad():
        loss = 0.0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            batch_loss = criterion(outputs,labels)
            _, preds = torch.max(outputs, 1)
            
            metric_recall.update(preds, labels)
            metric_precision.update(preds, labels)
            metric_accuracy.update(preds, labels)
            metric_f1.update(preds, labels)
            metric_confusion.update(preds, labels)
            loss += batch_loss.item()

        recall = metric_recall.compute().item()
        precision = metric_precision.compute().item()
        accuracy = metric_accuracy.compute().item()
        f1_score = metric_f1.compute().item()
        confusion_matrix = metric_confusion.compute().cpu()
        mean_loss = loss / len(loader) 

    per_class_metrics = compute_per_class_metrics(confusion_matrix, class_names)
    confusion_path = save_confusion_matrix(confusion_matrix, class_names, run_paths["confusion_matrix_path"])
    lowest_recall_classes = sorted(per_class_metrics, key=lambda item: item["recall"])[:5]

    print(f"<TEST>\tmean_loss={mean_loss}\taccuracy={accuracy:.4f}\trecall={recall:.4f}\tprecision={precision:.4f}\tf1={f1_score:.4f}")

    run_paths["reports_dir"].mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_loss": round(float(mean_loss), 6),
        "test_accuracy": round(float(accuracy), 6),
        "test_recall": round(float(recall), 6),
        "test_precision": round(float(precision), 6),
        "test_f1-score": round(float(f1_score), 6),
        "num_classes": num_classes,
        "random_baseline_accuracy": round(float(random_baseline), 6),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_best_val_accuracy": round(float(checkpoint.get("best_val_accuracy", 0.0)), 6),
    }

    with open(run_paths["metrics_path"], "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    per_class_report = {
        "arch_name": ARCH_NAME,
        "num_classes": num_classes,
        "per_class_metrics": per_class_metrics,
    }
    with open(run_paths["per_class_metrics_path"], "w", encoding="utf-8") as per_class_file:
        json.dump(per_class_report, per_class_file, indent=2)

    print("=== Evaluation Results on Test Set ===")
    print(f"Test Loss: {mean_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test F1-Score: {f1_score:.4f}")
    print("Lowest recall classes:")
    for row in lowest_recall_classes:
        print(
            f"- {row['class_name']}: "
            f"recall={row['recall']:.4f}, "
            f"precision={row['precision']:.4f}, "
            f"f1={row['f1_score']:.4f}, "
            f"support={row['support']}"
        )

    print(f"Saved metrics: {run_paths['metrics_path']}")
    print(f"Saved per-class metrics: {run_paths['per_class_metrics_path']}")
    print(f"Saved confusion matrix: {confusion_path}")

if __name__ == "__main__":
    evaluate_model()