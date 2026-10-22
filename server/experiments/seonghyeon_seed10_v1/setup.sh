#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_PY="${BASE_PY:-/home/sppark/miniconda3/envs/komap/bin/python}"
[[ -x "$EXPERIMENT_DIR/.venv/bin/python" ]] || "$BASE_PY" -m venv --system-site-packages "$EXPERIMENT_DIR/.venv"
"$EXPERIMENT_DIR/.venv/bin/python" -m pip install -r "$EXPERIMENT_DIR/requirements.txt"
NO_ALBUMENTATIONS_UPDATE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  "$EXPERIMENT_DIR/.venv/bin/python" -B -u "$EXPERIMENT_DIR/run.py" --prepare
