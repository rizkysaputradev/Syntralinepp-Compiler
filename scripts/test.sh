#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# SyntraLine++ test helper
#
# Usage:
#   ./scripts/test.sh
#
# This script:
#   - builds (Debug by default) if needed
#   - runs the C++ unit test binary: ./tests/syntra_tests
# -----------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

echo "[syntra-test] Root dir : ${ROOT_DIR}"
echo "[syntra-test] Build dir: ${BUILD_DIR}"

if [ ! -d "${BUILD_DIR}" ]; then
  echo "[syntra-test] Build dir not found; running build first..."
  "${ROOT_DIR}/scripts/build.sh" Debug
else
  # Ensure it's at least configured
  if [ ! -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    echo "[syntra-test] CMake not configured; running build..."
    "${ROOT_DIR}/scripts/build.sh" Debug
  fi
fi

cd "${BUILD_DIR}"

if [ ! -x "./tests/syntra_tests" ]; then
  echo "[syntra-test] syntra_tests binary missing; building..."
  cmake --build . --target syntra_tests -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

echo "[syntra-test] Running tests..."
./tests/syntra_tests
echo "[syntra-test] All tests completed."