#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
python -B "$EXPERIMENT_DIR/run.py" suite --gpus "${1:-8}"
