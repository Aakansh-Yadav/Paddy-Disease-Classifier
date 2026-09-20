"""Fine-tune an Ultralytics YOLO classification model on the paddy split."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.config import CLASSES, COMPARISON_PATH, MODELS_DIR, MODEL_REGISTRY, REPORTS_DIR
from src.dataset import load_metadata, stratified_split
from src.evaluate import save_confusion_matrix, save_predictions
from src.predict import ranking_from_scores
from src.yolo_data import export_classify_folder


def _device_arg() -> str:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def evaluate_yolo(model, val_df: pd.DataFrame) -> tuple[dict, list[int], list[int], list[dict]]:
    y_true, y_pred, records = [], [], []
    name_to_idx = {name: i for i, name in enumerate(CLASSES)}
    for row in val_df.itertuples(index=False):
        image = Image.open(row.path).convert("RGB")
        result = model.predict(source=image, verbose=False)[0]
        names = result.names
        probs = result.probs.data.cpu()
        pred_idx = int(probs.argmax().item())
        pred_name = names[pred_idx]
        true_name = row.label
        y_true.append(name_to_idx[true_name])
        y_pred.append(name_to_idx[pred_name])
        ranking = ranking_from_scores(
            {names[i]: float(probs[i]) for i in range(len(probs))}
        )
        records.append(
            {
                "image_id": row.image_id,
                "true_label": true_name,
                "predicted": pred_name,
                "confidence": ranking[0]["probability"],
                "second": ranking[1]["label"] if len(ranking) > 1 else "",
                "correct": pred_name == true_name,
            }
        )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": report,
    }
    return metrics, y_true, y_pred, records


def write_comparison() -> None:
    rows = []
    for key, spec in MODEL_REGISTRY.items():
        path = Path(spec["metrics"])
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows.append(
            {
                "model": key,
                "title": spec["title"],
                "val_accuracy": payload.get("val_accuracy"),
                "val_macro_f1": payload.get("val_macro_f1"),
                "n_train": payload.get("n_train"),
                "n_val": payload.get("n_val"),
                "val_errors": payload.get("val_errors"),
            }
        )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_PATH.write_text(json.dumps(rows, indent=2))


def train_yolo(
    model_key: str = "yolo11n-cls",
    epochs: int = 12,
    imgsz: int = 224,
    batch: int = 16,
) -> dict:
    from ultralytics import YOLO

    spec = MODEL_REGISTRY[model_key]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = export_classify_folder()
    _, val_df = stratified_split(load_metadata())
    model = YOLO(spec["pretrained"])
    model.train(
        data=str(data_dir),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=_device_arg(),
        project=str(MODELS_DIR),
        name=model_key.replace("-", "_"),
        exist_ok=True,
        pretrained=True,
        patience=8,
        verbose=True,
        workers=2,
    )
    best = MODELS_DIR / model_key.replace("-", "_") / "weights" / "best.pt"
    dest = Path(spec["checkpoint"])
    if not best.exists():
        raise FileNotFoundError(f"Ultralytics did not write {best}")
    shutil.copy2(best, dest)
    trained = YOLO(str(dest))
    metrics, y_true, y_pred, records = evaluate_yolo(trained, val_df)
    save_confusion_matrix(y_true, y_pred, path=spec["cm"])
    save_predictions(records, path=spec["errors"])
    payload = {
        "model": model_key,
        "val_accuracy": metrics["accuracy"],
        "val_macro_f1": metrics["macro_f1"],
        "n_train": int(sum(1 for _ in (data_dir / "train").rglob("*.jpg"))),
        "n_val": int(len(val_df)),
        "epochs": epochs,
        "classification_report": metrics["per_class"],
        "val_errors": int(sum(1 for row in records if not row["correct"])),
        "checkpoint": str(dest),
    }
    Path(spec["metrics"]).write_text(json.dumps(payload, indent=2))
    write_comparison()
    print(f"Saved YOLO checkpoint to {dest}")
    print(
        f"Val accuracy {payload['val_accuracy']:.3f}  "
        f"macro F1 {payload['val_macro_f1']:.3f}"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO-cls on the paddy split.")
    parser.add_argument(
        "--model",
        default="yolo11n-cls",
        choices=[k for k, v in MODEL_REGISTRY.items() if v["kind"] == "yolo"],
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    train_yolo(args.model, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)


if __name__ == "__main__":
    main()
