#!/usr/bin/env bash
# Mirror the code to a GPU server checkout. Set DSG_HOST (ssh host) and DSG_CODE_DIR (remote path).
# Generated schemas live in schemas/ on the server and come back with scripts/pull_from_server.sh.
set -euo pipefail
: "${DSG_HOST:?set DSG_HOST to the ssh host of the GPU server}"
: "${DSG_CODE_DIR:?set DSG_CODE_DIR to the code directory on the server}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
rsync -a --delete --exclude '._*' --exclude '.DS_Store' --exclude '__pycache__' --exclude '.git/' \
  --exclude 'report/' --exclude 'results/' --exclude 'schemas/' --exclude '*.zip' "$SRC/" "$DSG_HOST:$DSG_CODE_DIR/"
echo "synced $SRC -> $DSG_HOST:$DSG_CODE_DIR"
