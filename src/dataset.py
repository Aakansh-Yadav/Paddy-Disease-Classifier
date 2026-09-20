"""Load the Kaggle-style paddy dataset and build PyTorch dataloaders."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from src.config import (
    BATCH_SIZE,
    CLASS_TO_IDX,
    CLASSES,
    IMAGE_SIZE,
    NUM_WORKERS,
    SEED,
    TRAIN_CSV,
    TRAIN_IMAGES,
    VAL_FRACTION,
)


def image_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
                transforms.RandomCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_metadata(csv_path: Path = TRAIN_CSV) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run `python -m src.prepare_data` first, "
            "or set PADDY_DATA_DIR to your local paddy-disease-classification folder."
        )
    df = pd.read_csv(csv_path)
    required = {"image_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing columns: {sorted(missing)}")
    df["image_id"] = df["image_id"].astype(str)
    df["path"] = df.apply(
        lambda row: str(TRAIN_IMAGES / row["label"] / row["image_id"]), axis=1
    )
    exists = df["path"].map(lambda p: Path(p).exists())
    df = df.loc[exists].reset_index(drop=True)
    if df.empty:
        raise FileNotFoundError(
            f"No images found under {TRAIN_IMAGES}. Expected "
            "train_images/<label>/<image_id>.jpg"
        )
    unknown = sorted(set(df["label"]) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unexpected labels in train.csv: {unknown}")
    return df


class PaddyImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool):
        self.frame = frame.reset_index(drop=True)
        self.transform = image_transforms(train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = Image.open(row["path"]).convert("RGB")
        tensor = self.transform(image)
        label = CLASS_TO_IDX[row["label"]]
        return tensor, label, row["image_id"]


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df["label"].value_counts()
    too_small = counts[counts < 2]
    if not too_small.empty:
        raise ValueError(
            "Need at least 2 images per class for a stratified split. "
            f"Too few: {too_small.to_dict()}"
        )
    train_df, val_df = train_test_split(
        df,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def _weighted_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    counts = frame["label"].value_counts().to_dict()
    weights = frame["label"].map(lambda y: 1.0 / counts[y]).to_numpy()
    return WeightedRandomSampler(
        weights=weights, num_samples=len(frame), replacement=True
    )


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, pd.DataFrame, pd.DataFrame]:
    df = load_metadata()
    train_df, val_df = stratified_split(df)
    train_ds = PaddyImageDataset(train_df, train=True)
    val_ds = PaddyImageDataset(val_df, train=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=_weighted_sampler(train_df),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader, train_df, val_df
