"""Accuracy, precision, recall, and F1 for the 10-class validation split."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import CLASSES


def classification_scores(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
) -> dict:
    names = class_names or list(CLASSES)
    labels = list(range(len(names)))
    per_class = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
    }


def payload_from_scores(scores: dict) -> dict:
    """Flatten scores into the keys stored in reports/*.json."""
    return {
        "val_accuracy": scores["accuracy"],
        "val_precision": scores["precision_macro"],
        "val_recall": scores["recall_macro"],
        "val_macro_f1": scores["f1_macro"],
        "val_precision_weighted": scores["precision_weighted"],
        "val_recall_weighted": scores["recall_weighted"],
        "val_f1_weighted": scores["f1_weighted"],
        "classification_report": scores["per_class"],
    }


def headline_from_metrics(metrics: dict) -> dict:
    """Read accuracy / precision / recall / F1 from a saved metrics JSON."""
    report = metrics.get("classification_report") or {}
    macro = report.get("macro avg") or {}
    weighted = report.get("weighted avg") or {}
    accuracy = metrics.get("val_accuracy", report.get("accuracy"))
    return {
        "accuracy": float(accuracy) if accuracy is not None else None,
        "precision": float(metrics.get("val_precision") or macro.get("precision") or 0),
        "recall": float(metrics.get("val_recall") or macro.get("recall") or 0),
        "f1": float(metrics.get("val_macro_f1") or macro.get("f1-score") or 0),
        "precision_weighted": float(
            metrics.get("val_precision_weighted") or weighted.get("precision") or 0
        ),
        "recall_weighted": float(
            metrics.get("val_recall_weighted") or weighted.get("recall") or 0
        ),
        "f1_weighted": float(
            metrics.get("val_f1_weighted") or weighted.get("f1-score") or 0
        ),
    }
