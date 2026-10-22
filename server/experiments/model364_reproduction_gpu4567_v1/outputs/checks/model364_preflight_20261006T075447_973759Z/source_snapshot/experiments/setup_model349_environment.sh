#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${KOMAP_VENV:-$ROOT/.venv-model349}"
if [[ -e "$ENV_DIR" ]]; then
  echo "Refusing to overwrite existing environment: $ENV_DIR" >&2
  exit 2
fi
python3 -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install -r "$ROOT/requirements-model349-p100.txt"
echo "Environment created at $ENV_DIR"
echo "Run with: KOMAP_PYTHON=$ENV_DIR/bin/python KOMAP_DATA_DIR=/path/to/archives $ROOT/experiments/run_model349_pipeline.sh train"
