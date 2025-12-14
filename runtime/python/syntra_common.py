"""
SyntraLine++ common runtime utilities.

This module provides small helpers used by multiple backends (PyTorch / JAX),
such as:
- reproducible RNG seeding
- simple synthetic datasets when real CSV files are not available
- basic metric computation (accuracy, etc.)
"""

from __future__ import annotations

import os
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


@dataclass
class DatasetConfig:
    """
    Minimal dataset configuration mirroring SyntraLine++ ImageDataset fields.

    Fields:
        source:   path to CSV or data file (can be ignored by some runtimes)
        shape:    (H, W) for images (2D)
        channels: number of channels, e.g. 1 for grayscale
        batch:    batch size
        shuffle:  whether to shuffle batches
        seed:     integer seed for RNG
    """

    source: str
    shape: Tuple[int, int]
    channels: int
    batch: int
    shuffle: bool
    seed: int


@dataclass
class ModelConfig:
    """
    Minimal model configuration mirroring TorchModel fields.

    Fields:
        arch:       architecture identifier (e.g. "cnn", "cnn_small", "cnn_deep")
        framework:  "pytorch" or "jax" (currently informational)
        lr:         learning rate (float)
        epochs:     number of training epochs (int)
        optimizer:  optimizer name (e.g. "adam")
    """

    arch: str
    framework: str
    lr: float
    epochs: int
    optimizer: str


def set_global_seed(seed: int) -> None:
    """
    Set Python's global RNG to a deterministic seed.
    Backends using NumPy / torch / jax should also seed those separately.
    """
    random.seed(seed)


def compute_accuracy(logits, targets) -> float:
    """
    Compute accuracy given predictions and targets.

    This is intentionally backend-neutral: we accept either Python lists
    or tensor-like objects that support .argmax(-1) and equality ops.
    """

    # Best effort: try importing torch, else fallback to simple Python.
    try:
        import torch  # type: ignore

        if isinstance(logits, torch.Tensor):
            preds = logits.argmax(dim=-1)
            correct = (preds == targets).sum().item()
            total = targets.numel()
            return float(correct) / float(total) if total > 0 else 0.0

    except Exception:
        pass

    # Fallback: assume logits is a list of lists, targets a list[int]
    import math

    if not logits:
        return 0.0

    correct = 0
    total = 0
    for row, t in zip(logits, targets):
        if not row:
            continue
        # argmax in Python lists
        pred_idx = max(range(len(row)), key=lambda i: row[i])
        if pred_idx == t:
            correct += 1
        total += 1

    return float(correct) / float(total) if total > 0 else 0.0