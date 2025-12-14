#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# SyntraLine++ examples runner
#
# Usage:
#   ./scripts/run_examples.sh
#   ./scripts/run_examples.sh pytorch
#   ./scripts/run_examples.sh jax
#   ./scripts/run_examples.sh both
#
# This script:
#   - ensures the project is built
#   - iterates over all *.syntra files in ./examples
#   - runs `syntra --simulate` on each
#   - optionally emits PyTorch or JAX scripts for each example
# -----------------------------------------------------------------------------

MODE="${1:-simulate}"  # simulate / pytorch / jax / both

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
EXAMPLES_DIR="${ROOT_DIR}/examples"

echo "[syntra-examples] Root dir    : ${ROOT_DIR}"
echo "[syntra-examples] Build dir   : ${BUILD_DIR}"
echo "[syntra-examples] Examples dir: ${EXAMPLES_DIR}"
echo "[syntra-examples] Mode        : ${MODE}"

# Ensure build / syntra binary exists
if [ ! -d "${BUILD_DIR}" ] || [ ! -x "${BUILD_DIR}/src/syntra" ]; then
  echo "[syntra-examples] syntra binary not found; building..."
  "${ROOT_DIR}/scripts/build.sh" Debug
fi

cd "${BUILD_DIR}"

if [ ! -d "${EXAMPLES_DIR}" ]; then
  echo "[syntra-examples] No examples directory found at ${EXAMPLES_DIR}"
  exit 0
fi

shopt -s nullglob
EXAMPLE_FILES=( "${EXAMPLES_DIR}"/*.syntra )

if [ ${#EXAMPLE_FILES[@]} -eq 0 ]; then
  echo "[syntra-examples] No *.syntra files found in ${EXAMPLES_DIR}"
  exit 0
fi

for example in "${EXAMPLE_FILES[@]}"; do
  name="$(basename "${example}")"
  base="${name%.*}"

  echo
  echo "================================================================"
  echo "[syntra-examples] Example: ${name}"
  echo "================================================================"

  # simulate
  echo "[syntra-examples]  -> simulate"
  ./src/syntra "${example}" --simulate || {
    echo "[syntra-examples]  !! simulate failed for ${name}"
    continue
  }

  # emit pytorch
  if [ "${MODE}" = "pytorch" ] || [ "${MODE}" = "both" ]; then
    out_py="example_${base}_torch.py"
    echo "[syntra-examples]  -> emit-pytorch -> ${out_py}"
    ./src/syntra "${example}" --emit-pytorch > "${out_py}"
  fi

  # emit jax
  if [ "${MODE}" = "jax" ] || [ "${MODE}" = "both" ]; then
    out_py="example_${base}_jax.py"
    echo "[syntra-examples]  -> emit-jax -> ${out_py}"
    ./src/syntra "${example}" --emit-jax > "${out_py}"
  fi
done

echo
echo "[syntra-examples] Done."