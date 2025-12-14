#!/usr/bin/env python3
"""
make_mnist_csv.py

Download MNIST via torchvision and export it as two CSV files:

    data/mnist_train.csv
    data/mnist_test.csv

Each row has:
    label, p0, p1, ..., p783

where pixels are 0–255 (uint8), and the image is 28x28 in row-major order.
"""

import csv
from pathlib import Path

from torchvision.datasets import MNIST  # uses PIL internally


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def write_split(ds, out_path: Path, max_rows: int | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for img, label in ds:
            # img is a PIL.Image (by default), so we can use getdata()
            pixels = list(img.getdata())        # 28*28 values in [0, 255]
            row = [int(label)] + [int(p) for p in pixels]
            writer.writerow(row)
            count += 1
            if max_rows is not None and count >= max_rows:
                break

    print(f"[make_mnist_csv] Wrote {count} rows to {out_path}")


def main() -> None:
    print(f"[make_mnist_csv] Using data directory: {DATA_DIR}")

    train_ds = MNIST(root=DATA_DIR, train=True, download=True)
    test_ds  = MNIST(root=DATA_DIR, train=False, download=True)

    # keep it modest so CSVs are not huge
    MAX_TRAIN = 10000
    MAX_TEST  = 2000

    write_split(train_ds, DATA_DIR / "mnist_train.csv", max_rows=MAX_TRAIN)
    write_split(test_ds,  DATA_DIR / "mnist_test.csv",  max_rows=MAX_TEST)


if __name__ == "__main__":
    main()