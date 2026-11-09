"""Flag photos that do not look like Paddy Doctor leaf close-ups."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

LOW_CONFIDENCE = 0.45
HIGH_ENTROPY = 1.85  # nats; uniform over 10 classes is ln(10) ≈ 2.30
SKY_FRACTION = 0.12
WIDE_ASPECT = 1.45


def _entropy(scores: dict[str, float]) -> float:
    total = 0.0
    for value in scores.values():
        if value > 1e-12:
            total -= value * math.log(value)
    return total


def sky_fraction(image: Image.Image) -> float:
    """Share of the top quarter that looks like bright blue sky."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    height = max(int(rgb.shape[0] * 0.25), 1)
    top = rgb[:height]
    red, green, blue = top[:, :, 0], top[:, :, 1], top[:, :, 2]
    sky = (blue > 140) & (blue > green + 12) & (blue > red + 12)
    return float(sky.mean())


def assess_photo(image: Image.Image | None, scores: dict[str, float]) -> dict:
    """Return whether this still looks like a typical training leaf shot."""
    values = [float(v) for v in scores.values()]
    top = max(values) if values else 0.0
    entropy = _entropy(scores)
    aspect = 1.0
    sky = 0.0
    if image is not None:
        width, height = image.size
        aspect = width / max(height, 1)
        sky = sky_fraction(image)

    reasons: list[str] = []
    if top < LOW_CONFIDENCE:
        reasons.append(
            f"Top class confidence is only {top * 100:.0f}% (under {LOW_CONFIDENCE * 100:.0f}%)."
        )
    if entropy > HIGH_ENTROPY:
        reasons.append("The 10-class scores are spread out (high uncertainty).")
    if sky >= SKY_FRACTION:
        reasons.append(
            "A large blue region looks like sky — this is usually a field scene, not a leaf close-up."
        )
    if aspect >= WIDE_ASPECT and sky >= 0.05:
        reasons.append("The frame is wide and includes sky, unlike the training close-ups.")

    typical = len(reasons) == 0
    return {
        "typical_leaf_photo": typical,
        "top_confidence": top,
        "entropy": entropy,
        "sky_fraction": sky,
        "aspect": aspect,
        "reasons": reasons,
        "guidance": (
            "Use a close-up of one to three rice leaves in even daylight. "
            "Fill the frame with the leaf. Avoid sky, distant fields, and golden ripe panicles."
        ),
    }
