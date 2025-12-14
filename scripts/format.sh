#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# SyntraLine++ formatting helper
#
# Usage:
#   ./scripts/format.sh
#
# This script:
#   - runs clang-format in-place over:
#       - include/**/*.hpp
#       - src/**/*.cpp
# -----------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_DIR="${ROOT_DIR}/include"
SRC_DIR="${ROOT_DIR}/src"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "[syntra-format] clang-format not found in PATH."
  echo "[syntra-format] Install it (e.g., 'brew install clang-format') to enable formatting."
  exit 1
fi

echo "[syntra-format] Root dir     : ${ROOT_DIR}"
echo "[syntra-format] Include dir  : ${INCLUDE_DIR}"
echo "[syntra-format] Source dir   : ${SRC_DIR}"
echo "[syntra-format] Using clang-format: $(command -v clang-format)"

shopt -s globstar nullglob

# Format headers
HEADER_FILES=( "${INCLUDE_DIR}"/**/*.hpp )
if [ ${#HEADER_FILES[@]} -gt 0 ]; then
  echo "[syntra-format] Formatting headers..."
  clang-format -i "${HEADER_FILES[@]}"
fi

# Format sources
SOURCE_FILES=( "${SRC_DIR}"/**/*.cpp )
if [ ${#SOURCE_FILES[@]} -gt 0 ]; then
  echo "[syntra-format] Formatting sources..."
  clang-format -i "${SOURCE_FILES[@]}"
fi

echo "[syntra-format] Done."