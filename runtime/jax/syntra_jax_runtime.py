"""
SyntraLine++ JAX runtime.

This is a minimal JAX-based runtime that mirrors the behavior of the
inline JAX scripts currently produced by SyntraLine++: it creates
random image-like data, runs a tiny linear classifier or MLP, and
computes simple loss + accuracy metrics.

The goal is to provide a stable API that generated scripts can call:

    from jax.syntra_jax_runtime import run_experiment

    metrics = run_experiment(dataset_cfg, model_cfg, metrics=["accuracy", "loss"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import jax
import jax.numpy as jnp


@dataclass
class DatasetConfigJax:
    source: str
    shape: Tuple[int, int]
    channels: int
    batch: int
    shuffle: bool
    seed: int


@dataclass
class ModelConfigJax:
    arch: str
    framework: str
    lr: float
    epochs: int
    optimizer: str


def _init_params(rng_key, in_dim: int, num_classes: int, arch: str):
    """
    Initialize model parameters. For now we use a simple single-layer
    classifier regardless of 'arch', to keep the runtime light.
    """
    if arch in ("mlp", "cnn_deep", "cnn_grid"):
        # one hidden layer MLP
        k1, k2 = jax.random.split(rng_key)
        hidden_dim = 64
        W1 = jax.random.normal(k1, (in_dim, hidden_dim)) * 0.01
        b1 = jnp.zeros((hidden_dim,))
        W2 = jax.random.normal(k2, (hidden_dim, num_classes)) * 0.01
        b2 = jnp.zeros((num_classes,))
        return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    else:
        # simple linear classifier
        W = jax.random.normal(rng_key, (in_dim, num_classes)) * 0.01
        b = jnp.zeros((num_classes,))
        return {"W": W, "b": b}


def _forward(params, x, arch: str):
    if "W1" in params:
        # MLP
        h = jnp.dot(x, params["W1"]) + params["b1"]
        h = jax.nn.relu(h)
        logits = jnp.dot(h, params["W2"]) + params["b2"]
        return logits
    else:
        # Linear
        return jnp.dot(x, params["W"]) + params["b"]


def _loss_and_grad(params, x, y, arch: str):
    def loss_fn(p):
        logits = _forward(p, x, arch)
        one_hot = jax.nn.one_hot(y, logits.shape[-1])
        loss = -jnp.mean(jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1))
        return loss

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    return loss_val, grads


def run_experiment(
    dataset_cfg_dict: Dict[str, Any],
    model_cfg_dict: Dict[str, Any],
    metrics: List[str],
) -> Dict[str, float]:
    """
    Run a small JAX-based experiment on synthetic data.

    Returns:
        Dict[str, float]: at least 'loss' and 'accuracy'.
    """
    ds_cfg = DatasetConfigJax(
        source=str(dataset_cfg_dict.get("source", "")),
        shape=tuple(dataset_cfg_dict.get("shape", (28, 28))),
        channels=int(dataset_cfg_dict.get("channels", 1)),
        batch=int(dataset_cfg_dict.get("batch", 64)),
        shuffle=bool(dataset_cfg_dict.get("shuffle", True)),
        seed=int(dataset_cfg_dict.get("seed", 42)),
    )

    model_cfg = ModelConfigJax(
        arch=str(model_cfg_dict.get("arch", "cnn")),
        framework=str(model_cfg_dict.get("framework", "jax")),
        lr=float(model_cfg_dict.get("lr", 1e-3)),
        epochs=int(model_cfg_dict.get("epochs", 5)),
        optimizer=str(model_cfg_dict.get("optimizer", "adam")),
    )

    print(f"[Syntra-JAX] Running synthetic experiment (arch={model_cfg.arch})")

    rng = jax.random.PRNGKey(ds_cfg.seed)
    num_samples = 1024
    num_classes = 10
    h, w = ds_cfg.shape
    in_dim = ds_cfg.channels * h * w

    rng, kx, ky = jax.random.split(rng, 3)
    x = jax.random.normal(kx, (num_samples, in_dim))
    y = jax.random.randint(ky, (num_samples,), 0, num_classes)

    params = _init_params(rng, in_dim, num_classes, model_cfg.arch)

    @jax.jit
    def sgd_step(p, batch_x, batch_y, lr):
        loss_val, grads = _loss_and_grad(p, batch_x, batch_y, model_cfg.arch)
        new_p = jax.tree_util.tree_map(lambda w, g: w - lr * g, p, grads)
        return new_p, loss_val

    batch_size = ds_cfg.batch
    num_batches = num_samples // batch_size

    for epoch in range(1, model_cfg.epochs + 1):
        epoch_loss = 0.0
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            bx = x[start:end]
            by = y[start:end]
            params, loss_val = sgd_step(params, bx, by, model_cfg.lr)
            epoch_loss += float(loss_val)
        epoch_loss /= max(num_batches, 1)
        print(f"[Syntra-JAX] [epoch {epoch}/{model_cfg.epochs}] train loss = {epoch_loss:.4f}")

    # Evaluation on the same synthetic set
    logits = _forward(params, x, model_cfg.arch)
    preds = jnp.argmax(logits, axis=-1)
    correct = jnp.sum(preds == y)
    acc = float(correct) / float(num_samples)

    # Recompute loss
    one_hot = jax.nn.one_hot(y, num_classes)
    loss_vec = -jnp.sum(one_hot * jax.nn.log_softmax(logits), axis=-1)
    loss_val = float(jnp.mean(loss_vec))

    print(f"[Syntra-JAX] eval accuracy={acc:.4f}, loss={loss_val:.4f}")

    result: Dict[str, float] = {}
    for name in metrics:
        if name == "accuracy":
            result["accuracy"] = acc
        elif name == "loss":
            result["loss"] = loss_val

    if "accuracy" not in result:
        result["accuracy"] = acc
    if "loss" not in result:
        result["loss"] = loss_val

    return result