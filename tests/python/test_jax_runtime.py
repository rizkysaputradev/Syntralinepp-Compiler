"""
JAX runtime tests for SyntraLine++.

Again, we use synthetic data and only check that the runtime executes
and produces sane metrics (accuracy, loss).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, Any

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from runtime.jax.syntra_jax_runtime import run_experiment


def make_mnist_like_dataset_cfg() -> Dict[str, Any]:
    return {
        "source": "data/mnist_train.csv",
        "shape": (28, 28),
        "channels": 1,
        "batch": 64,
        "shuffle": True,
        "seed": 123,  # different seed from PyTorch test for coverage
    }


def make_basic_jax_model_cfg() -> Dict[str, Any]:
    return {
        "arch": "cnn",  # interpreted as linear/MLP in our JAX runtime
        "framework": "jax",
        "lr": 1e-3,
        "epochs": 2,    # keep tests quick
        "optimizer": "adam",
    }


@pytest.mark.cpu
def test_jax_runtime_runs_and_returns_metrics():
    dataset_cfg = make_mnist_like_dataset_cfg()
    model_cfg = make_basic_jax_model_cfg()

    metrics = ["accuracy", "loss"]
    result = run_experiment(dataset_cfg, model_cfg, metrics)

    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "loss" in result

    assert isinstance(result["accuracy"], float)
    assert isinstance(result["loss"], float)

    assert 0.0 <= result["accuracy"] <= 1.0
    assert math.isfinite(result["loss"])