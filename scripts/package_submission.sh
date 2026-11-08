#!/usr/bin/env bash
# Build <name>_code_project1.zip: code, configs, schemas and result tables only
# (no datasets, no models, no run files, no external links).
#   scripts/package_submission.sh <Name> <output dir>
set -euo pipefail
NAME=${1:?usage: package_submission.sh <Name> <output dir>}
OUT=${2:?usage: package_submission.sh <Name> <output dir>}
SRC="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)/${NAME}_code_project1"
mkdir -p "$STAGE"
rsync -a --exclude '._*' --exclude '.DS_Store' --exclude '__pycache__' --exclude '.git/' \
  --exclude 'report/' --exclude '*.parquet' --exclude '*.zip' --exclude 'data/' --exclude 'runs/' \
  --exclude 'cache/' --exclude 'logs/' "$SRC/" "$STAGE/"
if grep -rIn -E 'https?://' "$STAGE" | grep -v '127\.0\.0\.1' | grep -q .; then
  echo "external links found:"; grep -rIn -E 'https?://' "$STAGE" | grep -v '127\.0\.0\.1'; exit 1
fi
mkdir -p "$OUT"
(cd "$(dirname "$STAGE")" && COPYFILE_DISABLE=1 zip -qr -X "$OUT/${NAME}_code_project1.zip" "$(basename "$STAGE")")
echo "wrote $OUT/${NAME}_code_project1.zip ($(du -h "$OUT/${NAME}_code_project1.zip" | cut -f1))"
