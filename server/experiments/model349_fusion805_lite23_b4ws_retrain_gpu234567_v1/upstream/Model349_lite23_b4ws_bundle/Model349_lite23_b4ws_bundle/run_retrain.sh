#!/usr/bin/env bash
# Retrain all 23 weights for one seed and predict valid (scored) / test with the fixed lite23 + B4ws config.
#   bash run_retrain.sh SEED DATA OUT [GPUS e.g. 0,1,2,3]
# DATA: train/{images,masks} (70), valid/{images,masks} (20), test/images (optional). ~90 P100 GPU-hours per seed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
bash "$HERE/Model349_fusion805_lite23_b4ws/retrain/retrain_all.sh" "$1" "$HERE/Model349_fusion805" "$HERE/train_route" "$2" "$3" "${4:-0}"
