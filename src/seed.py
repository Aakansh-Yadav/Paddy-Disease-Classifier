"""Single fixed seed for splits, loaders, MobileNet, and YOLO."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from src.config import SEED


def set_seed(seed: int = SEED) -> int:
    """Pin Python, NumPy, and PyTorch RNGs. Do not retune this to chase val scores."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, RuntimeError):
        pass
    return seed


def torch_generator(seed: int = SEED) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
