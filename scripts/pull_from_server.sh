#!/usr/bin/env bash
# Pull generated schemas and aggregated results (small files only) back from the GPU server.
set -euo pipefail
: "${DSG_HOST:?set DSG_HOST to the ssh host of the GPU server}"
: "${DSG_CODE_DIR:?set DSG_CODE_DIR to the code directory on the server}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$SRC/schemas" "$SRC/results"
rsync -a --exclude '._*' "$DSG_HOST:$DSG_CODE_DIR/schemas/" "$SRC/schemas/"
rsync -a --exclude '._*' --exclude '*.parquet' "$DSG_HOST:$DSG_CODE_DIR/results/" "$SRC/results/" 2>/dev/null || true
echo "pulled schemas and results into $SRC"
