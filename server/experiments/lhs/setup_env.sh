#!/usr/bin/env bash
set -euo pipefail
LHS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LHS_BASE_PYTHON="${LHS_BASE_PYTHON:-/home/sppark/miniconda3/envs/komap/bin/python}"
if [[ ! -x "$LHS_DIR/.venv/bin/python" ]]; then
  "$LHS_BASE_PYTHON" -m venv --system-site-packages "$LHS_DIR/.venv"
fi
"$LHS_DIR/.venv/bin/python" -m pip install -r "$LHS_DIR/requirements.txt"
"$LHS_DIR/.venv/bin/python" -m pip check
"$LHS_DIR/run.sh" prepare326
"$LHS_DIR/run.sh" check
