"""Grad-CAM heatmap so the app can show which leaf regions drove the prediction."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torch import nn

from src.dataset import image_transforms


def target_layer(model: nn.Module) -> nn.Module:
    return model.features[-1]


def _colorize(cam: np.ndarray) -> Image.Image:
    colored = cm.jet(cam)[..., :3]
    return Image.fromarray((colored * 255).astype(np.uint8))


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = _colorize(cam).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heat.convert("RGB"), alpha)


def compute_gradcam(
    model: nn.Module,
    image: Image.Image,
    class_idx: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    tensor = image_transforms(train=False)(image.convert("RGB")).unsqueeze(0).to(device)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def on_forward(_module, _inp, out):
        activations.append(out)

    def on_backward(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = target_layer(model)
    h_fwd = layer.register_forward_hook(on_forward)
    h_bwd = layer.register_full_backward_hook(on_backward)
    try:
        logits = model(tensor)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    acts = activations[0].detach()[0]
    grads = gradients[0].detach()[0]
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam.cpu().numpy()
    cam -= cam.min()
    peak = cam.max()
    if peak > 0:
        cam /= peak
    return cam
