"""Download a working subset or verify a local Kaggle-style dataset folder."""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.config import CLASSES, DATA_DIR, SUBSET_PER_CLASS, TRAIN_CSV, TRAIN_IMAGES


def _write_csv(rows: list[dict]) -> None:
    TRAIN_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TRAIN_CSV, index=False)


def _rows_from_disk() -> list[dict]:
    rows: list[dict] = []
    existing: dict[str, dict] = {}
    if TRAIN_CSV.exists():
        for row in pd.read_csv(TRAIN_CSV).to_dict(orient="records"):
            existing[str(row.get("image_id"))] = row
    for label in CLASSES:
        folder = TRAIN_IMAGES / label
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            meta = existing.get(path.name, {})
            rows.append(
                {
                    "image_id": path.name,
                    "label": label,
                    "variety": meta.get("variety", "unknown"),
                    "age": meta.get("age", 0),
                }
            )
    return rows


def class_counts() -> dict[str, int]:
    counts = {label: 0 for label in CLASSES}
    for row in _rows_from_disk():
        counts[row["label"]] += 1
    return counts


def dataset_ready(min_images: int = 20) -> bool:
    return sum(class_counts().values()) >= min_images


def download_huggingface_subset(per_class: int) -> int:
    from datasets import load_dataset

    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    collected: dict[str, list[dict]] = defaultdict(list)
    for row in _rows_from_disk():
        collected[row["label"]].append(row)

    if all(len(collected[c]) >= per_class for c in CLASSES):
        rows = [row for items in collected.values() for row in items]
        _write_csv(rows)
        print(f"Already have {len(rows)} images (>= {per_class} per class).")
        return len(rows)

    print("Streaming Hugging Face dataset anthony2261/paddy-disease-classification ...")
    ds = load_dataset(
        "anthony2261/paddy-disease-classification",
        split="train",
        streaming=True,
    )
    try:
        for example in ds:
            label = example["label"]
            if isinstance(label, int):
                label = CLASSES[label]
            if label not in CLASSES or len(collected[label]) >= per_class:
                if all(len(collected[c]) >= per_class for c in CLASSES):
                    break
                continue
            image = example["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")
            else:
                image = image.convert("RGB")
            image_id = f"{label}_{len(collected[label]):03d}.jpg"
            out_dir = TRAIN_IMAGES / label
            out_dir.mkdir(parents=True, exist_ok=True)
            dest = out_dir / image_id
            if not dest.exists():
                image.save(dest, quality=90)
            collected[label].append(
                {
                    "image_id": image_id,
                    "label": label,
                    "variety": example.get("variety", "unknown"),
                    "age": example.get("age", 0),
                }
            )
            filled = sum(len(v) for v in collected.values())
            if filled % 20 == 0:
                print(f"  saved {filled} images")
    finally:
        del ds

    rows = [row for items in collected.values() for row in items]
    if not rows:
        raise RuntimeError("Hugging Face download returned no images.")
    _write_csv(rows)
    print(f"Wrote {len(rows)} images to {TRAIN_IMAGES}")
    return len(rows)


def _draw_leaf(path: Path, label: str, index: int) -> None:
    palettes = {
        "bacterial_leaf_blight": ((34, 120, 52), (210, 190, 70)),
        "bacterial_leaf_streak": ((40, 130, 55), (240, 230, 140)),
        "bacterial_panicle_blight": ((70, 110, 40), (150, 90, 50)),
        "blast": ((36, 110, 48), (90, 50, 30)),
        "brown_spot": ((50, 120, 45), (120, 70, 30)),
        "dead_heart": ((90, 100, 40), (180, 170, 120)),
        "downy_mildew": ((80, 140, 90), (220, 230, 180)),
        "hispa": ((45, 125, 50), (250, 250, 250)),
        "normal": ((28, 130, 48), (40, 160, 60)),
        "tungro": ((50, 120, 40), (230, 150, 40)),
    }
    base, mark = palettes[label]
    img = Image.new("RGB", (256, 256), (210, 225, 190))
    draw = ImageDraw.Draw(img)
    draw.ellipse((40, 20, 216, 240), fill=base)
    for i in range(6):
        x0 = 50 + (i * 23 + index * 7) % 140
        y0 = 40 + (i * 29 + index * 11) % 150
        draw.ellipse((x0, y0, x0 + 28, y0 + 18), fill=mark)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((12, 8), f"{label}\n#{index}", fill=(20, 30, 20), font=font)
    img.save(path, quality=90)


def write_synthetic_fallback(per_class: int = 24) -> int:
    print("Could not download the public dataset. Writing synthetic demo images.")
    rows: list[dict] = []
    for label in CLASSES:
        out_dir = TRAIN_IMAGES / label
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            image_id = f"synth_{label}_{i:03d}.jpg"
            _draw_leaf(out_dir / image_id, label, i)
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "variety": "ADT45",
                    "age": 45,
                }
            )
    _write_csv(rows)
    print(f"Wrote {len(rows)} synthetic images to {TRAIN_IMAGES}")
    return len(rows)


def _label_name(example, features=None) -> str:
    label = example["label"]
    if isinstance(label, str) and label in CLASSES:
        return label
    if features is not None and hasattr(features.get("label"), "int2str"):
        name = features["label"].int2str(int(label))
        if name in CLASSES:
            return name
    if isinstance(label, int) and 0 <= label < len(CLASSES):
        return CLASSES[label]
    raise ValueError(f"Unknown label: {label!r}")


def download_huggingface_full() -> int:
    """Download every train image from the public Paddy Doctor mirror (~10,407)."""
    from datasets import load_dataset

    if TRAIN_IMAGES.exists():
        shutil.rmtree(TRAIN_IMAGES)
    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    for label in CLASSES:
        (TRAIN_IMAGES / label).mkdir(parents=True, exist_ok=True)

    print("Downloading full Hugging Face set anthony2261/paddy-disease-classification ...")
    ds = load_dataset("anthony2261/paddy-disease-classification", split="train")
    features = getattr(ds, "features", None)
    counts: dict[str, int] = defaultdict(int)
    rows: list[dict] = []
    try:
        for i, example in enumerate(ds):
            label = _label_name(example, features)
            image = example["image"]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")
            else:
                image = image.convert("RGB")
            image_id = f"{label}_{counts[label]:05d}.jpg"
            image.save(TRAIN_IMAGES / label / image_id, quality=90)
            counts[label] += 1
            rows.append(
                {
                    "image_id": image_id,
                    "label": label,
                    "variety": example.get("variety", "unknown"),
                    "age": example.get("age", 0),
                }
            )
            if (i + 1) % 500 == 0:
                print(f"  saved {i + 1} images")
    finally:
        del ds

    if len(rows) < 10000:
        raise RuntimeError(f"Expected ~10407 images, got {len(rows)}")
    _write_csv(rows)
    print(f"Wrote {len(rows)} images to {TRAIN_IMAGES}")
    print("Per class:", dict(counts))
    return len(rows)


def prepare(
    per_class: int,
    force_download: bool,
    synthetic: bool,
    full: bool = False,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    counts = class_counts()
    total = sum(counts.values())
    if full:
        if total >= 10000 and not force_download:
            print(f"Full dataset already present: {total} rows at {DATA_DIR}")
            return
        download_huggingface_full()
        return
    enough = all(counts[c] >= per_class for c in CLASSES)
    if dataset_ready() and not force_download and enough:
        print(f"Dataset already present: {total} rows at {DATA_DIR}")
        return
    if synthetic:
        write_synthetic_fallback(per_class)
        return
    try:
        download_huggingface_subset(per_class)
    except Exception as exc:
        print(f"Download failed ({exc})")
        write_synthetic_fallback(per_class)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the paddy disease dataset.")
    parser.add_argument("--per-class", type=int, default=SUBSET_PER_CLASS)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the entire ~10,407-image train set (not an 80-per-class subset).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip download and generate colored demo images.",
    )
    args = parser.parse_args()
    prepare(args.per_class, args.force_download, args.synthetic, full=args.full)


if __name__ == "__main__":
    main()
    # Hugging Face streaming can abort on interpreter shutdown after a good run.
    import os

    os._exit(0)
