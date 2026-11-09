"""Validation metrics, confusion matrix, and per-image error table."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from src.config import (
    CHECKPOINT,
    CLASSES,
    CM_PATH,
    ERRORS_PATH,
    IDX_TO_CLASS,
    METRICS_PATH,
    REPORTS_DIR,
)
from src.dataset import create_dataloaders
from src.metrics import classification_scores, payload_from_scores
from src.model import load_checkpoint


def evaluate_loader(model, loader, device):
    model.eval()
    y_true, y_pred, records = [], [], []
    with torch.no_grad():
        for images, labels, ids in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            confidences, seconds = torch.topk(probs, k=2, dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            for i, image_id in enumerate(ids):
                true_idx = int(labels[i])
                pred_idx = int(preds[i].cpu())
                records.append(
                    {
                        "image_id": image_id,
                        "true_label": IDX_TO_CLASS[true_idx],
                        "predicted": IDX_TO_CLASS[pred_idx],
                        "confidence": float(confidences[i, 0].cpu()),
                        "second": IDX_TO_CLASS[int(seconds[i, 1].cpu())],
                        "correct": true_idx == pred_idx,
                    }
                )
    scores = classification_scores(y_true, y_pred)
    return (
        {
            "accuracy": scores["accuracy"],
            "macro_f1": scores["f1_macro"],
            "precision_macro": scores["precision_macro"],
            "recall_macro": scores["recall_macro"],
            "per_class": scores["per_class"],
            **payload_from_scores(scores),
        },
        y_true,
        y_pred,
        records,
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


def save_predictions(records: list[dict], path=ERRORS_PATH) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Train a model first. Missing {CHECKPOINT}")
    model, _payload = load_checkpoint(CHECKPOINT, device)
    _train_loader, val_loader, _train_df, _val_df = create_dataloaders()
    report, y_true, y_pred, records = evaluate_loader(model, val_loader, device)
    save_confusion_matrix(y_true, y_pred)
    save_predictions(records)
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    else:
        metrics = {}
    metrics["val_accuracy"] = report["accuracy"]
    metrics["val_precision"] = report["precision_macro"]
    metrics["val_recall"] = report["recall_macro"]
    metrics["val_macro_f1"] = report["macro_f1"]
    metrics["classification_report"] = report["per_class"]
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(
        json.dumps(
            {
                "accuracy": report["accuracy"],
                "precision": report["precision_macro"],
                "recall": report["recall_macro"],
                "macro_f1": report["macro_f1"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
