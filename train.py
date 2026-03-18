import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score, Precision, Recall

from src.classifier import CardClassifier
from src.utils import (
    plot_metric_curves,
    set_seed,
    train_loader_utils,
    val_loader_utils,
)
from config import (
    SEED,
    EPOCH_NUM,
    DEVICE,
    ARCH_NAME,
    SHOW_PLOTS,
    IMAGE_SIZE,
    USE_PRETRAINED_WEIGHTS,
    get_run_paths,
)


def train_model():
    set_seed(SEED)

    run_paths = get_run_paths(
        arch_name=ARCH_NAME,
        use_pretrained_weights=USE_PRETRAINED_WEIGHTS,
    )

    train_loader = train_loader_utils(image_size=IMAGE_SIZE)
    val_loader = val_loader_utils(image_size=IMAGE_SIZE)

    num_classes = len(train_loader.dataset.classes)
    if len(val_loader.dataset.classes) != num_classes:
        raise ValueError("Train and validation splits have different number of classes.")
    if train_loader.dataset.classes != val_loader.dataset.classes:
        raise ValueError("Train and validation splits have different class-name ordering.")

    device = torch.device(DEVICE)
    print(f"Using: {DEVICE}")
    print(f"Architecture: {ARCH_NAME}")
    print(f"Classes: {num_classes}")
    print(f"Run: {run_paths['run_name']}")

    net = CardClassifier(
        arch_name=ARCH_NAME,
        num_classes=num_classes,
        pretrained=USE_PRETRAINED_WEIGHTS,
    )
    net = net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-2)

    train_metric_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    train_metric_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    train_metric_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    train_metric_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)

    val_metric_recall = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    val_metric_precision = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
    val_metric_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_metric_f1 = F1Score(task="multiclass", num_classes=num_classes, average="weighted").to(device)

    loss = []
    recall = []
    precision = []
    accuracy = []
    f1_score = []

    best_val_accuracy = float("-inf")
    best_epoch = -1
    best_checkpoint_path = run_paths["best_checkpoint_path"]

    for epoch in range(EPOCH_NUM):
        train_loss = 0.0
        val_loss = 0.0

        net.train()
        for image, label in train_loader:
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = net(image)
            batch_loss = criterion(outputs, label)
            _, preds = torch.max(outputs, 1)
            batch_loss.backward()
            optimizer.step()

            train_metric_recall.update(preds, label)
            train_metric_precision.update(preds, label)
            train_metric_accuracy.update(preds, label)
            train_metric_f1.update(preds, label)
            train_loss += batch_loss.item()

        net.eval()
        with torch.no_grad():
            for image, label in val_loader:
                image, label = image.to(device), label.to(device)
                outputs = net(image)
                batch_loss = criterion(outputs, label)
                _, preds = torch.max(outputs, 1)

                val_metric_recall.update(preds, label)
                val_metric_precision.update(preds, label)
                val_metric_accuracy.update(preds, label)
                val_metric_f1.update(preds, label)
                val_loss += batch_loss.item()

        # Training
        train_recall = train_metric_recall.compute()
        train_precision = train_metric_precision.compute()
        train_accuracy = train_metric_accuracy.compute()
        train_f1 = train_metric_f1.compute()

        train_metric_recall.reset()
        train_metric_precision.reset()
        train_metric_accuracy.reset()
        train_metric_f1.reset()
        
        print(f"<TRN>\tepoch={epoch}\taccuracy={train_accuracy:.4f}\trecall={train_recall:.4f}\tprecision={train_precision:.4f}\tf1={train_f1:.4f}")


        # Validation
        val_recall = val_metric_recall.compute()
        val_precision = val_metric_precision.compute()
        val_accuracy = val_metric_accuracy.compute()
        val_f1 = val_metric_f1.compute()
        
        val_metric_recall.reset()
        val_metric_precision.reset()
        val_metric_accuracy.reset()
        val_metric_f1.reset()
        
        print(f"<VAL>\tepoch={epoch}\taccuracy={val_accuracy:.4f}\trecall={val_recall:.4f}\tprecision={val_precision:.4f}\tf1={val_f1:.4f}")

        # Summarize metrics
        mean_train_loss = train_loss / len(train_loader)
        mean_val_loss = val_loss / len(val_loader)
        loss.append([mean_train_loss, mean_val_loss])

        recall.append([train_recall, val_recall])
        precision.append([train_precision, val_precision])
        accuracy.append([train_accuracy, val_accuracy])
        f1_score.append([train_f1, val_f1])

        print(f"mean_train_loss={mean_train_loss:.4f}\t mean_val_loss={mean_val_loss:.4f}\n")

        current_val_accuracy = float(val_accuracy.detach().cpu())
        if current_val_accuracy > best_val_accuracy:
            best_val_accuracy = current_val_accuracy
            best_epoch = epoch
            best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                "arch_name": ARCH_NAME,
                "use_pretrained_weights": USE_PRETRAINED_WEIGHTS,
                "image_size": IMAGE_SIZE,
                "num_classes": num_classes,
                "class_names": train_loader.dataset.classes,
                "class_to_idx": train_loader.dataset.class_to_idx,
                "epoch": epoch,
                "best_val_accuracy": best_val_accuracy,
                "best_val_loss": mean_val_loss,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(
                f"Saved new best checkpoint: {best_checkpoint_path} "
                f"(epoch={epoch}, val_accuracy={best_val_accuracy:.4f})"
            )

    plot_metric_curves(data=loss, title_label="Training and Validation Loss Over Epochs", curve_label="Loss", y_label="Loss Value", show_plot=SHOW_PLOTS, filename="loss_curve.png", output_dir=run_paths["figures_dir"])
    plot_metric_curves(data=recall, title_label="Training and Validation Recall Over Epochs", curve_label="Recall", y_label="Recall Value", show_plot=SHOW_PLOTS, filename="recall_curve.png", output_dir=run_paths["figures_dir"])
    plot_metric_curves(data=precision, title_label="Training and Validation Precision Over Epochs", curve_label="Precision", y_label="Precision Value", show_plot=SHOW_PLOTS, filename="precision_curve.png", output_dir=run_paths["figures_dir"])
    plot_metric_curves(data=accuracy, title_label="Training and Validation Accuracy Over Epochs", curve_label="Accuracy", y_label="Accuracy Value", show_plot=SHOW_PLOTS, filename="accuracy_curve.png", output_dir=run_paths["figures_dir"])
    plot_metric_curves(data=f1_score, title_label="Training and Validation F1-Score Over Epochs", curve_label="F1-Score ", y_label="F1-Score Value", show_plot=SHOW_PLOTS, filename="f1_score_curve.png", output_dir=run_paths["figures_dir"])

    print(
        f"Best checkpoint summary: epoch={best_epoch}, "
        f"val_accuracy={best_val_accuracy:.4f}, path={best_checkpoint_path}"
    )

if __name__ == "__main__":
    train_model()
