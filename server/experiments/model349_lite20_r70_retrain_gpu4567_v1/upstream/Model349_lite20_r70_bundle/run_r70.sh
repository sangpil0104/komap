#!/usr/bin/env bash
# lite20_r70: lite20 whose R B4 paths (322/326/329) are one train70 model each instead of valid-selected folds.
# Trains the 3 B4 models, then predicts valid20 (scored) and test10 with lite20_r70 AND lite20 on the same other weights,
# and writes a paired comparison.
#   bash run_r70.sh SEED DATA R_WEIGHTS ASSETS OUT [GPUS e.g. 0,1,2] [--allow-retrained]
# DATA      : train/{images,masks} (70), valid/{images,masks} (20), test/images (optional)
# R_WEIGHTS : R checkpoints in the exh14 layout   - published: WEIGHTS/R          - seed S retrain: runs/seedS/R/weights
# ASSETS    : E/F checkpoints                       - published: WEIGHTS/assets     - seed S retrain: runs/seedS/assets
# --allow-retrained : needed when R_WEIGHTS/ASSETS come from retrain (skips SHA checks). The 3 new B4 are never SHA-checked.
# SEED only sets the B4 training seed (0 + 1000*SEED). Use the same SEED as the retrain run you pair it with.
set -euo pipefail
S=$1; DATA=$(cd "$2" && pwd); RW=$(cd "$3" && pwd); AS=$(cd "$4" && pwd); OUT=$5; GPUS=${6:-0}; EXTRA=${7:-}
HERE=$(cd "$(dirname "$0")" && pwd); PY=${PYTHON:-python}; L=$HERE/Model349_fusion805_lite23_b4ws
mkdir -p "$OUT/logs"; OUT=$(cd "$OUT" && pwd); IFS=, read -ra G <<< "$GPUS"
# 1) train
bash "$HERE/r70/train_r70.sh" "$S" "$DATA" "$OUT/b4_full70" "$GPUS" 2>&1 | tee "$OUT/logs/train_r70.log"
grep -q "R70 B4 DONE" "$OUT/logs/train_r70.log" || exit 1
# 2) assemble: R weights = given R weights + the 3 train70 B4 (links only; nothing is copied or modified)
mkdir -p "$OUT/weights_R"
for e in "$RW"/*; do ln -sfn "$e" "$OUT/weights_R/$(basename "$e")"; done
for d in "$OUT/b4_full70"/*_full70; do ln -sfn "$d" "$OUT/weights_R/$(basename "$d")"; done
# code-only copy of the parent package whose Model349_exh14/weights points at weights_R
if [[ ! -e $OUT/parent ]]; then
  mkdir -p "$OUT/parent"; (cd "$HERE/Model349_fusion805" && tar --exclude=./Model349_exh14/weights -cf - .) | (cd "$OUT/parent" && tar -xf -)
  ln -s "$OUT/weights_R" "$OUT/parent/Model349_exh14/weights"
fi
# 3) predict (first GPU; infer_lite checks it is idle at start)
for v in lite20_r70 lite20; do for split in valid test; do
  [[ -d $DATA/$split/images ]] || continue; d=$OUT/$v/$split; [[ -e $d/completion.json ]] && continue; rm -rf "$d"; mkdir -p "$OUT/$v"
  "$PY" "$L/infer_compact.py" --config "$L/selected_config_$v.json" --parent-package "$OUT/parent" --assets "$AS" \
    --input "$DATA/$split/images" --output "$d" --gpus "${G[0]}" $EXTRA > "$OUT/logs/${v}_$split.log" 2>&1
done
[[ -d $DATA/valid/masks ]] && bash "$HERE/score.sh" "$OUT/$v/valid/fusion/predicted_labels" "$DATA/valid/masks" "$OUT/$v/valid_metrics.json"
done
# 4) paired comparison
"$PY" "$HERE/r70/compare_r70.py" "$OUT" | tee "$OUT/summary.txt"
