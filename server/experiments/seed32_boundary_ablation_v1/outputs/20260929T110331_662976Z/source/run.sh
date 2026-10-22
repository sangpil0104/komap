#!/usr/bin/env bash
set -euo pipefail
ABLATION_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export NO_ALBUMENTATIONS_UPDATE=1
exec conda run -n komap --no-capture-output python -B -u "$ABLATION_DIR/run.py" "$@"
