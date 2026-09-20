"""Build an Ultralytics classify folder (train/val/<class>/*.jpg) from the same split."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from src.config import CLASSES, YOLO_DATA_DIR
from src.dataset import load_metadata, stratified_split


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def export_classify_folder(root: Path = YOLO_DATA_DIR) -> Path:
    """Write train/ and val/ class folders using the project's stratified split."""
    if root.exists():
        shutil.rmtree(root)
    df = load_metadata()
    train_df, val_df = stratified_split(df)
    for split_name, frame in (("train", train_df), ("val", val_df)):
        for label in CLASSES:
            (root / split_name / label).mkdir(parents=True, exist_ok=True)
        for row in frame.itertuples(index=False):
            src = Path(row.path)
            dest = root / split_name / row.label / src.name
            _link_or_copy(src, dest)
    print(
        f"YOLO classify data: {len(train_df)} train / {len(val_df)} val images at {root}"
    )
    return root
