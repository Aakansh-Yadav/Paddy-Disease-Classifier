"""Exploratory plots for the paddy disease dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image

from src.config import CLASS_DIST_PATH, CLASSES, REPORTS_DIR, SAMPLES_PATH
from src.dataset import load_metadata


def plot_class_distribution(df: pd.DataFrame, path: Path = CLASS_DIST_PATH) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    order = [c for c in CLASSES if c in set(df["label"])]
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, y="label", order=order, color="#2e7d32")
    plt.title("Training images per class")
    plt.xlabel("Count")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_sample_grid(df: pd.DataFrame, path: Path = SAMPLES_PATH) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
    for ax, label in zip(axes.ravel(), CLASSES):
        subset = df[df["label"] == label]
        ax.set_title(label.replace("_", " "), fontsize=8)
        ax.axis("off")
        if subset.empty:
            continue
        image = Image.open(subset.iloc[0]["path"]).convert("RGB")
        ax.imshow(image)
    fig.suptitle("One example per class", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_eda() -> dict:
    df = load_metadata()
    plot_class_distribution(df)
    plot_sample_grid(df)
    summary = {
        "n_images": int(len(df)),
        "n_classes": int(df["label"].nunique()),
        "class_counts": df["label"].value_counts().to_dict(),
    }
    if "variety" in df.columns:
        summary["varieties"] = df["variety"].value_counts().head(10).to_dict()
    if "age" in df.columns:
        summary["age_mean"] = float(pd.to_numeric(df["age"], errors="coerce").mean())
    print(f"EDA complete. {summary['n_images']} images, plots in {REPORTS_DIR}")
    return summary


if __name__ == "__main__":
    run_eda()
