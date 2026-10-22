#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec conda run --no-capture-output -n komap python -B "$EXPERIMENT_DIR/run.py" suite --gpus "${1:-8}"
