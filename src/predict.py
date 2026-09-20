"""Single-image inference for MobileNet and YOLO-cls backends."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.config import CHECKPOINT, CLASS_INFO, CLASSES, CLASS_TO_IDX, MODEL_REGISTRY
from src.dataset import image_transforms
from src.model import load_checkpoint


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ranking_from_scores(scores: dict[str, float], topk: int = 5) -> list[dict]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:topk]
    ranking = []
    for name, score in ordered:
        ranking.append(
            {
                "label": name,
                "probability": float(score),
                "index": CLASS_TO_IDX[name],
                **CLASS_INFO[name],
            }
        )
    return ranking


def pack_result(ranking: list[dict], backend: str) -> dict:
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
        "backend": backend,
    }


def predict_torch(
    image: Image.Image,
    model,
    device: torch.device,
    topk: int = 5,
) -> dict:
    tensor = image_transforms(train=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    scores = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    return pack_result(ranking_from_scores(scores, topk=topk), backend="mobilenet_v3_small")


def predict_yolo(image: Image.Image, model, topk: int = 5, backend: str = "yolo11n-cls") -> dict:
    result = model.predict(source=image, verbose=False)[0]
    names = result.names
    probs = result.probs.data.cpu()
    scores = {names[i]: float(probs[i]) for i in range(len(probs)) if names[i] in CLASS_TO_IDX}
    return pack_result(ranking_from_scores(scores, topk=topk), backend=backend)


def load_backend(key: str):
    spec = MODEL_REGISTRY[key]
    ckpt = Path(spec["checkpoint"])
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No weights for {spec['title']} at {ckpt}. Train it first."
        )
    device = get_device()
    if spec["kind"] == "yolo":
        from ultralytics import YOLO

        return YOLO(str(ckpt)), device, spec
    model, payload = load_checkpoint(ckpt, device)
    return model, device, spec


def available_backends() -> list[str]:
    return [key for key, spec in MODEL_REGISTRY.items() if Path(spec["checkpoint"]).exists()]


def predict_image(
    image: Image.Image | str | Path,
    model=None,
    device: torch.device | None = None,
    topk: int = 5,
    backend: str = "mobilenet_v3_small",
) -> dict:
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    spec = MODEL_REGISTRY[backend]
    if model is None:
        model, device, spec = load_backend(backend)
    if spec["kind"] == "yolo":
        return predict_yolo(image, model, topk=topk, backend=backend)
    device = device or get_device()
    return predict_torch(image, model, device, topk=topk)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--backend",
        default="mobilenet_v3_small",
        choices=list(MODEL_REGISTRY),
    )
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image, backend=args.backend), indent=2))
