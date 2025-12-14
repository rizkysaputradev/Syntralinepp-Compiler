#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[syntra-runtime-test] Root: ${ROOT}"

cd "${ROOT}"

# If your dedicated venv exists, use it; otherwise, just run with whatever python is active.
if [ -d ".syntra-venv" ]; then
    echo "[syntra-runtime-test] Activating .syntra-venv"
    # shellcheck source=/dev/null
    source .syntra-venv/bin/activate
else
    echo "[syntra-runtime-test] .syntra-venv not found, using current Python environment"
fi

# Make sure pytest is installed (if you used extras/dev in pyproject, you can skip this).
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[syntra-runtime-test] pytest not found, installing..."
    pip install pytest
fi

echo "[syntra-runtime-test] Running pytest on tests/python..."
pytest -q tests/python