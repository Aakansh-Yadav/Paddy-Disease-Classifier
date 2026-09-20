"""Classify every image in a folder and write a CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

from src.config import CHECKPOINT, REPORTS_DIR
from src.model import load_checkpoint
from src.predict import get_device, predict_image


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            yield path


def batch_predict(folder: Path, output: Path) -> pd.DataFrame:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Train a model first. Missing {CHECKPOINT}")
    device = get_device()
    model, _ = load_checkpoint(CHECKPOINT, device)
    rows = []
    paths = list(iter_images(folder))
    if not paths:
        raise FileNotFoundError(f"No images under {folder}")
    for path in paths:
        image = Image.open(path).convert("RGB")
        result = predict_image(image, model=model, device=device)
        rows.append(
            {
                "path": str(path),
                "image_id": path.name,
                "predicted": result["label"],
                "confidence": round(result["probability"], 4),
                "second": result["ranking"][1]["label"] if len(result["ranking"]) > 1 else "",
                "second_confidence": (
                    round(result["ranking"][1]["probability"], 4)
                    if len(result["ranking"]) > 1
                    else ""
                ),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    print(f"Wrote {len(frame)} predictions to {output}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-classify a folder of leaf photos.")
    parser.add_argument("folder")
    parser.add_argument(
        "-o",
        "--output",
        default=str(REPORTS_DIR / "batch_predictions.csv"),
    )
    args = parser.parse_args()
    batch_predict(Path(args.folder), Path(args.output))


if __name__ == "__main__":
    main()
