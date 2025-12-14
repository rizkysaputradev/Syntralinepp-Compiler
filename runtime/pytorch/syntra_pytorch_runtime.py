"""
SyntraLine++ PyTorch runtime.

This module implements a simple end-to-end training + evaluation routine
for image-like classification experiments, roughly matching the inline
scripts that SyntraLine++ currently generates.

The idea is that in the future, generated scripts become very thin:

    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "runtime"))
    from pytorch.syntra_pytorch_runtime import run_experiment

    dataset_cfg = {...}
    model_cfg = {...}
    metrics = run_experiment(dataset_cfg, model_cfg, metrics=["accuracy", "loss"])
    
    print("Final experiment result:")
    print(metrics)

For now, this module is "backend API ready" even if C++ still inlines
parts of the script.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from runtime.python.syntra_common import DatasetConfig, ModelConfig, set_global_seed


# -----------------------------------------------------------------------------
# Dataset: synthetic image classification (MNIST-like)
# -----------------------------------------------------------------------------


class RandomImageDataset(Dataset):
    """
    Simple synthetic dataset: random "images" with random labels.

    Used as a fallback when a real CSV / dataset backend is not implemented.
    """

    def __init__(
        self,
        num_samples: int,
        shape: Tuple[int, int],
        channels: int,
        num_classes: int,
        seed: int,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.shape = shape
        self.channels = channels
        self.num_classes = num_classes

        g = torch.Generator()
        g.manual_seed(seed)

        c, h, w = channels, shape[0], shape[1]
        self.images = torch.randn(num_samples, c, h, w, generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def make_dataloaders(cfg: DatasetConfig, num_samples: int = 1024):
    """
    Create train/test DataLoaders from a DatasetConfig.

    Right now we ignore cfg.source and always create random data.
    In the future, this can be extended to load real CSV / datasets.
    """
    set_global_seed(cfg.seed)

    # 80/20 split
    train_n = int(num_samples * 0.8)
    test_n = num_samples - train_n

    train_ds = RandomImageDataset(
        num_samples=train_n,
        shape=cfg.shape,
        channels=cfg.channels,
        num_classes=10,
        seed=cfg.seed,
    )
    test_ds = RandomImageDataset(
        num_samples=test_n,
        shape=cfg.shape,
        channels=cfg.channels,
        num_classes=10,
        seed=cfg.seed + 1,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch,
        shuffle=cfg.shuffle,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch,
        shuffle=False,
    )
    return train_loader, test_loader


# -----------------------------------------------------------------------------
# Simple CNN model family
# -----------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """
    Very small CNN for 28x28 images, parameterized by a string 'arch'.
    """

    def __init__(self, in_channels: int, num_classes: int, arch: str = "cnn") -> None:
        super().__init__()

        if arch in ("cnn_small", "cnn"):
            hidden = 32
        elif arch in ("cnn_deep", "cnn_grid"):
            hidden = 64
        else:
            hidden = 32  # default

        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(hidden * 7 * 7 if arch.startswith("cnn") else hidden * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_model(model_cfg: ModelConfig, in_channels: int, num_classes: int) -> nn.Module:
    """
    Construct a model from a ModelConfig.
    """
    arch = model_cfg.arch or "cnn"
    model = SimpleCNN(in_channels=in_channels, num_classes=num_classes, arch=arch)
    return model


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    total_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        total_loss += loss.item()
        total_batches += 1

        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    avg_loss = total_loss / max(total_batches, 1)
    acc = float(correct) / float(total) if total > 0 else 0.0
    return {"loss": avg_loss, "accuracy": acc}


def run_experiment(
    dataset_cfg_dict: Dict[str, Any],
    model_cfg_dict: Dict[str, Any],
    metrics: List[str],
) -> Dict[str, float]:
    """
    Run a single image classification experiment using the given configs.

    Args:
        dataset_cfg_dict: dictionary with keys like "source", "shape", "channels",
                          "batch", "shuffle", "seed".
        model_cfg_dict:   dictionary with keys like "arch", "framework", "lr",
                          "epochs", "optimizer".
        metrics:          list of metric names; we currently support "loss", "accuracy".

    Returns:
        Dict[str, float]: at least 'loss' and 'accuracy'.
    """
    ds_cfg = DatasetConfig(
        source=dataset_cfg_dict.get("source", ""),
        shape=tuple(dataset_cfg_dict.get("shape", (28, 28))),
        channels=int(dataset_cfg_dict.get("channels", 1)),
        batch=int(dataset_cfg_dict.get("batch", 64)),
        shuffle=bool(dataset_cfg_dict.get("shuffle", True)),
        seed=int(dataset_cfg_dict.get("seed", 42)),
    )

    model_cfg = ModelConfig(
        arch=str(model_cfg_dict.get("arch", "cnn")),
        framework=str(model_cfg_dict.get("framework", "pytorch")),
        lr=float(model_cfg_dict.get("lr", 1e-3)),
        epochs=int(model_cfg_dict.get("epochs", 5)),
        optimizer=str(model_cfg_dict.get("optimizer", "adam")),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_dataloaders(ds_cfg, num_samples=1024)

    model = make_model(model_cfg, in_channels=ds_cfg.channels, num_classes=10)
    model.to(device)

    if model_cfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=model_cfg.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg.lr)

    for epoch in range(1, model_cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, device, optimizer)
        print(f"[epoch {epoch}/{model_cfg.epochs}] train loss = {train_loss:.4f}")

    stats = evaluate(model, test_loader, device)
    print(f"eval stats: {stats}")

    # Filter metrics dict to requested keys
    result: Dict[str, float] = {}
    for name in metrics:
        if name in stats:
            result[name] = float(stats[name])

    # Ensure at least "loss" and "accuracy" are present if known
    for k in ("loss", "accuracy"):
        if k in stats and k not in result:
            result[k] = float(stats[k])

    return result