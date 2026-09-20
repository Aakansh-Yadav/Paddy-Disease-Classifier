"""Train the paddy disease classifier."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from tqdm import tqdm

from src.config import (
    CHECKPOINT,
    EPOCHS,
    HISTORY_PATH,
    LEARNING_RATE,
    METRICS_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    WEIGHT_DECAY,
)
from src.dataset import create_dataloaders
from src.eda import run_eda
from src.evaluate import evaluate_loader, save_confusion_matrix
from src.model import build_model, save_checkpoint


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    losses = []
    y_true, y_pred = [], []
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _ids in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def plot_history(history: list[dict], path: Path = HISTORY_PATH) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train loss")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Val loss")
    plt.plot(epochs, [row["val_accuracy"] for row in history], label="Val accuracy")
    plt.xlabel("Epoch")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def train(
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    skip_eda: bool = False,
) -> dict:
    set_seed()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not skip_eda:
        run_eda()

    device = get_device()
    train_loader, val_loader, train_df, val_df = create_dataloaders()
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_f1 = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs}  "
            f"train_acc={row['train_accuracy']:.3f}  "
            f"val_acc={row['val_accuracy']:.3f}  "
            f"val_f1={row['val_macro_f1']:.3f}"
        )
        if val_metrics["macro_f1"] >= best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    report, y_true, y_pred = evaluate_loader(model, val_loader, device)
    save_confusion_matrix(y_true, y_pred)
    plot_history(history)
    payload = {
        "val_accuracy": report["accuracy"],
        "val_macro_f1": report["macro_f1"],
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "epochs": epochs,
        "device": str(device),
        "history": history,
        "classification_report": report["per_class"],
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    save_checkpoint(model, CHECKPOINT, extra={"metrics": payload})
    print(f"Saved checkpoint to {CHECKPOINT}")
    print(f"Val accuracy {payload['val_accuracy']:.3f}  macro F1 {payload['val_macro_f1']:.3f}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--skip-eda", action="store_true")
    args = parser.parse_args()
    train(epochs=args.epochs, lr=args.lr, skip_eda=args.skip_eda)


if __name__ == "__main__":
    main()
