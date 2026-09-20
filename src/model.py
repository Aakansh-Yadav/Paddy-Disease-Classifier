"""MobileNetV3-Small classifier for the ten paddy disease classes."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.config import CLASSES, PRETRAINED


def build_model(pretrained: bool = PRETRAINED) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASSES))
    return model


def save_checkpoint(model: nn.Module, path, extra: dict | None = None) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "classes": CLASSES,
        **(extra or {}),
    }
    torch.save(payload, path)


def load_checkpoint(path, device: torch.device) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload
