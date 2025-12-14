#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# SyntraLine++ build helper
#
# Usage:
#   ./scripts/build.sh            # default: Debug
#   ./scripts/build.sh Debug      # explicit Debug
#   ./scripts/build.sh Release    # Release build
#
# This script:
#   - creates (or reuses) ./build
#   - configures CMake with the chosen build type
#   - builds the 'syntra' and 'syntra_tests' targets
# -----------------------------------------------------------------------------

BUILD_TYPE="${1:-Debug}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "[syntra-build] Root dir : ${ROOT_DIR}"
echo "[syntra-build] Build dir: ${BUILD_DIR}"
echo "[syntra-build] Build type: ${BUILD_TYPE}"

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure CMake (idempotent)
echo "[syntra-build] Configuring CMake..."
cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" ..

# Build all targets
echo "[syntra-build] Building targets..."
cmake --build . -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "[syntra-build] Done."