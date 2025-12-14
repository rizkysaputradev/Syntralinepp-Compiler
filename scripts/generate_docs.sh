#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# SyntraLine++ documentation helper
#
# Usage:
#   ./scripts/generate_docs.sh
#
# This is a placeholder for now. In the future, you might:
#   - run Doxygen over the C++ headers
#   - convert docs/*.md to HTML or PDF
#   - generate API references
# -----------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS_DIR="${ROOT_DIR}/docs"

echo "[syntra-docs] Root dir : ${ROOT_DIR}"
echo "[syntra-docs] Docs dir : ${DOCS_DIR}"

if [ ! -d "${DOCS_DIR}" ]; then
  echo "[syntra-docs] Docs directory does not exist yet."
  exit 0
fi

echo "[syntra-docs] Currently this script is a stub."
echo "[syntra-docs] You can wire it to Doxygen / Sphinx / pandoc in the future."