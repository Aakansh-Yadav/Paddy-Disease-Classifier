"""Grad-CAM heatmap so the app can show which leaf regions drove the prediction."""

from __future__ import annotations

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from torch import nn

from src.dataset import image_transforms


def last_conv2d(model: nn.Module) -> nn.Module:
    layer = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            layer = module
    if layer is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return layer


def target_layer(model: nn.Module) -> nn.Module:
    if hasattr(model, "features"):
        return model.features[-1]
    return last_conv2d(model)


def _colorize(cam: np.ndarray) -> Image.Image:
    colored = cm.jet(cam)[..., :3]
    return Image.fromarray((colored * 255).astype(np.uint8))


def overlay_cam(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = _colorize(cam).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heat.convert("RGB"), alpha)


def _as_logits(output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("logits", next(iter(output.values())))
    return output


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
        activations.append(out if torch.is_tensor(out) else out[0])

    def on_backward(_module, _grad_in, grad_out):
        gradients.append(grad_out[0])

    layer = target_layer(model)
    h_fwd = layer.register_forward_hook(on_forward)
    h_bwd = layer.register_full_backward_hook(on_backward)
    try:
        logits = _as_logits(model(tensor))
        if logits.ndim > 2:
            logits = logits.flatten(1)
        model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        h_fwd.remove()
        h_bwd.remove()

    acts = activations[0].detach()[0]
    grads = gradients[0].detach()[0]
    if acts.ndim != 3:
        raise ValueError("Grad-CAM needs a CHW feature map.")
    weights = grads.mean(dim=(1, 2))
    cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = cam.cpu().numpy()
    cam -= cam.min()
    peak = cam.max()
    if peak > 0:
        cam /= peak
    return cam


def maybe_heatmap(backend_key: str, loaded_model, image: Image.Image, class_idx: int, device):
    from src.config import MODEL_REGISTRY

    spec = MODEL_REGISTRY[backend_key]
    if not spec.get("gradcam"):
        return None
    torch_model = loaded_model.model if spec["kind"] == "yolo" else loaded_model
    try:
        cam = compute_gradcam(torch_model, image, class_idx, device)
        return overlay_cam(image, cam)
    except Exception:
        return None
