#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
K=${KOMAP_ROOT:-$HOME/komap}
PY=${PYTHON:-/home/sangpil/miniforge3/envs/komap/bin/python}
if [[ ${1:-} == --help || $# -lt 2 ]]; then
 echo "Usage: $0 INPUT_IMAGES NEW_OUTPUT [GPU=0] [VERIFIED_R_RELEASE]"
 [[ ${1:-} == --help ]] && exit 0
 exit 2
fi
EXTRA=()
if [[ -n ${4:-} ]]; then EXTRA=(--release "$4"); fi
exec "$PY" "$HERE/infer_compact.py" --config "$HERE/selected_config.json" \
 --parent-package "$K/Model349_lite20_b4ws_bundle/Model349_fusion805" --assets "$HERE/assets" \
 --input "$1" --output "$2" --gpus "${3:-0}" "${EXTRA[@]}"
