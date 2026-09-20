"""Validation metrics and confusion matrix."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import CHECKPOINT, CLASSES, CM_PATH, METRICS_PATH, REPORTS_DIR
from src.dataset import create_dataloaders
from src.model import load_checkpoint


def evaluate_loader(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels, _ids in loader:
            logits = model(images.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    labels = list(range(len(CLASSES)))
    per_class = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    return (
        {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "per_class": per_class,
        },
        y_true,
        y_pred,
    )


def save_confusion_matrix(y_true, y_pred, path=CM_PATH) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))
    plt.figure(figsize=(8.5, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=[c.replace("_", "\n") for c in CLASSES],
        yticklabels=[c.replace("_", " ") for c in CLASSES],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Validation confusion matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Train a model first. Missing {CHECKPOINT}")
    model, _payload = load_checkpoint(CHECKPOINT, device)
    _train_loader, val_loader, _train_df, _val_df = create_dataloaders()
    report, y_true, y_pred = evaluate_loader(model, val_loader, device)
    save_confusion_matrix(y_true, y_pred)
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    else:
        metrics = {}
    metrics.update(report)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"accuracy": report["accuracy"], "macro_f1": report["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
