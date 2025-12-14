#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[syntra-integration-test] Root: ${ROOT}"

cd "${ROOT}"

# Activate .syntra-venv if present (reuse same environment as runtime tests)
if [ -d ".syntra-venv" ]; then
    echo "[syntra-integration-test] Activating .syntra-venv"
    # shellcheck source=/dev/null
    source .syntra-venv/bin/activate
else
    echo "[syntra-integration-test] .syntra-venv not found, using current Python environment"
fi

# Ensure the C++ binary exists; if not, build it.
if [ ! -x "build/src/syntra" ]; then
    echo "[syntra-integration-test] syntra binary not found, running ./scripts/build.sh..."
    ./scripts/build.sh
fi

# Make sure pytest is installed (if you rely on pyproject/requirements, this should already be true)
if ! python -c "import pytest" >/dev/null 2>&1; then
    echo "[syntra-integration-test] pytest not found, installing..."
    pip install pytest
fi

echo "[syntra-integration-test] Running pytest on tests/integration..."
pytest -q tests/integration
