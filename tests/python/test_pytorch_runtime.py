"""
PyTorch runtime tests for SyntraLine++.

These are high-level "does it run and return sane metrics?" tests,
not correctness tests against real MNIST (we use synthetic data).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Any

import pytest

# Ensure we can import the runtime package when running tests directly.
# If you've done `pip install -e .` this is not strictly necessary,
# but it's a nice fallback when just running `pytest` from the repo.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from runtime.pytorch.syntra_pytorch_runtime import run_experiment


def make_mnist_like_dataset_cfg() -> Dict[str, Any]:
    # Mirrors the mnist_basic.syntra dataset block semantically
    return {
        "source": "data/mnist_train.csv",  # currently ignored by runtime
        "shape": (28, 28),
        "channels": 1,
        "batch": 64,
        "shuffle": True,
        "seed": 42,
    }


def make_basic_cnn_model_cfg() -> Dict[str, Any]:
    # Mirrors a simple TorchModel in your examples
    return {
        "arch": "cnn",
        "framework": "pytorch",
        "lr": 1e-3,
        "epochs": 2,      # keep tests fast
        "optimizer": "adam",
    }


@pytest.mark.cpu
def test_pytorch_runtime_runs_and_returns_metrics():
    dataset_cfg = make_mnist_like_dataset_cfg()
    model_cfg = make_basic_cnn_model_cfg()

    metrics = ["accuracy", "loss"]
    result = run_experiment(dataset_cfg, model_cfg, metrics)

    # Basic structural checks
    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "loss" in result

    # Type sanity
    assert isinstance(result["accuracy"], float)
    assert isinstance(result["loss"], float)

    # Range sanity: accuracy in [0, 1], loss finite
    assert 0.0 <= result["accuracy"] <= 1.0
    assert math.isfinite(result["loss"])