#!/usr/bin/env bash
set -euo pipefail
# ANA: input images, new output directory, physical GPU (default 0), optional verified R release.
HERE=$(cd "$(dirname "$0")" && pwd)
K=$HOME/komap
PY=${PYTHON:-$K/../miniforge3/envs/komap/bin/python}
EXTRA=()
if [[ -n ${4:-} ]]; then EXTRA=(--release "$4"); fi
exec "$PY" "$HERE/infer_compact.py" --config "$HERE/selected_config_lite16_b413.json" \
  --parent-package "$K/Model349_lite20_b4ws_bundle/Model349_fusion805" --assets "$K/weights/all/assets" \
  --input "$1" --output "$2" --gpus "${3:-0}" "${EXTRA[@]}"
