#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 DATA_ROOT" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA_ROOT="$(cd "$1" && pwd -P)"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/model403_local_supcon}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:?Set DEVICE to a P100 device that is available for this run}"
if [[ -L "$ROOT/outputs" || -L "$OUT_ROOT" ]]; then
  echo "Refusing symlinked outputs path: $ROOT/outputs or $OUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"
if [[ -L "$OUT_ROOT" ]]; then
  echo "Refusing symlinked Model403 output root: $OUT_ROOT" >&2
  exit 3
fi
OUT_ROOT="$(cd "$OUT_ROOT" && pwd -P)"
SPLITS="$OUT_ROOT/splits"
BASELINE_ROOT="$ROOT/outputs/model392_widthnorm"
TRAINER="$ROOT/experiments/model403_local_supcon_train.py"
EVALUATOR="$ROOT/model397_reproduction_bundle/model392/evaluate_image_mean.py"
WEIGHT_LOCK="$ROOT/experiments/results/model403_fold_specific_weight_lock.json"
LOCK_FILE="$OUT_ROOT/.model403_oof.lock"

if [[ -L "$SPLITS" || -L "$OUT_ROOT/results" || -L "$LOCK_FILE" ]]; then
  echo "Refusing symlinked Model403 output child/lock path." >&2
  exit 3
fi
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another Model403 OOF runner already holds $LOCK_FILE." >&2
  exit 6
fi

if [[ -e "$OUT_ROOT/results/model403_oof.json" || -e "$OUT_ROOT/results/model403_oof.json.claim" ]]; then
  echo "Model403 OOF score already exists/was claimed; preserving it and refusing a rerun." >&2
  exit 3
fi
if [[ -L "$BASELINE_ROOT" ]]; then
  echo "Refusing symlinked baseline output root: $BASELINE_ROOT" >&2
  exit 3
fi
for fold in 1 2 3 4; do
  baseline="$BASELINE_ROOT/seed1/fold${fold}_oof/valid_probabilities.npz"
  if [[ ! -s "$baseline" ]]; then
    echo "Missing locked Model392 seed1 OOF archive: $baseline" >&2
    exit 4
  fi
  if [[ -L "$OUT_ROOT/fold$fold" || -L "$OUT_ROOT/fold${fold}_oof" ]]; then
    echo "Refusing symlinked Model403 fold output for fold $fold." >&2
    exit 3
  fi
done
if [[ ! -s "$WEIGHT_LOCK" ]]; then
  echo "Missing per-fold outer-train-only weight lock: $WEIGHT_LOCK" >&2
  exit 4
fi

"$PYTHON" "$ROOT/model397_reproduction_bundle/scripts/prepare_oof_splits.py" \
  --data-root "$DATA_ROOT" --output-root "$SPLITS"

for fold in 1 2 3 4; do
  data="$SPLITS/model392_cv/seed1_fold$fold"
  run="$OUT_ROOT/fold$fold"
  supcon_weight="$("$PYTHON" - "$WEIGHT_LOCK" "$ROOT" "$fold" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

lock_path, root_path, fold = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
lock = json.loads(lock_path.read_text(encoding="utf-8"))
entry = lock["folds"][fold]
source = root_path / entry["calibration_path"]
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if digest != entry["calibration_sha256"]:
    raise SystemExit(f"calibration source hash mismatch for fold {fold}: {source}")
expected = float(lock["probe_weight"]) / float(entry["median_gradient_norm_ratio"])
weight = float(entry["supcon_weight"])
if not math.isclose(weight, expected, rel_tol=0.0, abs_tol=1e-15):
    raise SystemExit(f"locked weight does not follow train-only calibration for fold {fold}")
print(f"{weight:.15g}")
PY
)"
  if [[ -f "$run/last.pt" && -f "$run/train_state.pt" ]]; then
    completed_epoch="$("$PYTHON" - "$run/train_state.pt" "$supcon_weight" <<'PY'
import math
import sys
import torch

state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
args = state["args"]
if int(args["seed"]) != 1 or not math.isclose(
    float(args["supcon_weight"]), float(sys.argv[2]), rel_tol=0.0, abs_tol=1e-15
):
    raise SystemExit("existing checkpoint does not match this fold's locked recipe")
print(int(state["epoch"]))
PY
)"
    if [[ "$completed_epoch" -ge 70 ]]; then
      echo "fold $fold already completed epoch $completed_epoch; preserving its checkpoint."
    else
      "$PYTHON" "$TRAINER" --data-root "$data" --output-dir "$run" \
        --device "$DEVICE" --seed 1 --supcon-weight "$supcon_weight" --resume
    fi
  elif [[ -e "$run" ]]; then
    echo "Incomplete or non-resumable output exists at $run; preserve it and select a new OUT_ROOT." >&2
    exit 3
  else
    mkdir -p "$run"
    "$PYTHON" "$TRAINER" --data-root "$data" --output-dir "$run" \
      --device "$DEVICE" --seed 1 --supcon-weight "$supcon_weight"
  fi

  oof="$OUT_ROOT/fold${fold}_oof"
  if [[ -f "$oof/valid_probabilities.npz" ]]; then
    echo "fold $fold OOF probabilities already exist; preserving them."
  elif [[ -e "$oof" ]]; then
    echo "Incomplete OOF output exists at $oof; refusing to overwrite." >&2
    exit 3
  else
    "$PYTHON" "$EVALUATOR" --checkpoint "$run/last.pt" --data-root "$data" \
      --split valid --output-dir "$oof" --device "$DEVICE" --batch-size 2 \
      --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4
  fi
done

"$PYTHON" "$ROOT/experiments/score_model403_supcon_oof.py" \
  --data-root "$DATA_ROOT" --output-root "$OUT_ROOT" \
  --baseline-root "$BASELINE_ROOT" --out "$OUT_ROOT/results/model403_oof.json"
