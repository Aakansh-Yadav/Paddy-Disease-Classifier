"""Single-image inference for the trained classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.config import CHECKPOINT, CLASS_INFO, CLASSES
from src.dataset import image_transforms
from src.model import load_checkpoint


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_image(
    image: Image.Image | str | Path,
    model=None,
    device: torch.device | None = None,
    topk: int = 5,
) -> dict:
    device = device or get_device()
    if model is None:
        if not CHECKPOINT.exists():
            raise FileNotFoundError(
                "No trained model found. Run `python -m src.train` first."
            )
        model, _ = load_checkpoint(CHECKPOINT, device)
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    tensor = image_transforms(train=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probs, k=min(topk, len(CLASSES)))
    ranking = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        name = CLASSES[idx]
        ranking.append(
            {
                "label": name,
                "probability": score,
                "index": idx,
                **CLASS_INFO[name],
            }
        )
    top = ranking[0]
    return {
        "label": top["label"],
        "title": top["title"],
        "probability": top["probability"],
        "index": top["index"],
        "cause": top["cause"],
        "signs": top["signs"],
        "action": top["action"],
        "ranking": ranking,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image), indent=2))
