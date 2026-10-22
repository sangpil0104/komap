#!/usr/bin/env bash
# Official metric (image-wise 4-class mIoU, empty union = 0):  bash score.sh PRED_LABELS_DIR GT_MASKS_DIR OUT_JSON
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
${PYTHON:-python} "$HERE/Model349_fusion805/score_masks.py" --labels "$1" --gt "$2" --output "$3"
${PYTHON:-python} -c "import json,sys;print('mIoU', json.load(open(sys.argv[1]))['image_miou']*100)" "$3"
