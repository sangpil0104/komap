#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
CODE="$ROOT/model336_hrnet_best"
OUT="$ROOT/outputs/model377_hrnet_full_train_seeds"
GPU_ID=6
MODEL376_RESULT="$ROOT/outputs/model376_bagged_hrnet_valid_eval/results.json"

[[ -x "$PY" && -d "$CODE" && -d "$ROOT/data/train/images" ]] || {
  echo "missing Python, Model336 code, or full training split" >&2
  exit 2
}
[[ -s "$MODEL376_RESULT" ]] || {
  echo "Model377 fallback requires the completed Model376 one-shot result: $MODEL376_RESULT" >&2
  exit 2
}
"$PY" - "$MODEL376_RESULT" <<'PY'
import json, math, sys
from pathlib import Path
result = json.loads(Path(sys.argv[1]).read_text())
if (result.get("model") != 376
        or result.get("metric") != "20-image mean of four class IoUs; empty union=0"):
    raise SystemExit("Model376 result does not use the official metric")
score = float(result["candidate_image_mean_miou_empty0"])
if not math.isfinite(score) or score >= 0.80:
    raise SystemExit(f"Model377 fallback is predeclared only for Model376 < 0.80; got {score}")
print(f"Model376 official score {score:.8f} is below 0.80; Model377 fallback allowed")
PY
mkdir -p "$OUT"
"$PY" - "$ROOT/data" <<'PY'
import sys
from pathlib import Path
root = Path(sys.argv[1])
images = list((root / "train/images").glob("*_image.png"))
masks = list((root / "train/masks").glob("*_mask.png"))
if len(images) != 70 or len(masks) != 70:
    raise SystemExit(f"Model377 expects 70 training images/masks; got {len(images)}/{len(masks)}")
PY

for seed in 1 2; do
  case "$seed" in 1|2) ;; *) echo "invalid Model377 seed: $seed" >&2; exit 2 ;; esac
  run="$OUT/seed$seed"
  mkdir -p "$run"
  if [[ -s "$run/args.json" ]]; then
    "$PY" - "$run/args.json" "$ROOT/data" "$run" "$seed" <<'PY'
import json, sys
from pathlib import Path
args_path, data_root, output_dir = map(Path, sys.argv[1:4])
seed = int(sys.argv[4])
args = json.loads(args_path.read_text())
expected = {
    "data_root": str(data_root.resolve()), "output_dir": str(output_dir.resolve()),
    "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
    "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
    "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
    "num_workers": 4, "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
    "weight_decay": 1e-4, "ce_class_weights": None, "init_checkpoint": None,
    "no_boundary_head": False, "boundary_head_weight": 1.0,
    "seed": seed, "val_every": 0, "no_amp": False, "resume": True,
}
if any(args.get(key) != value for key, value in expected.items()):
    raise SystemExit(f"existing seed-{seed} args.json differs from the fixed Model377 recipe")
PY
  elif [[ -s "$run/train_state.pt" || -s "$run/last.pt" ]]; then
    echo "seed $seed checkpoint/state exists without args.json: $run" >&2
    exit 3
  fi
  state_epoch=0
  if [[ -s "$run/train_state.pt" ]]; then
    state_epoch=$("$PY" - "$run/train_state.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state["epoch"]))
PY
)
  elif [[ -s "$run/last.pt" ]]; then
    echo "checkpoint exists without resumable train_state: $run" >&2
    exit 3
  fi
  if (( state_epoch < 70 )); then
    echo "=== Model377 full-train HRNet seed $seed: fixed 70 epochs, no valid checkpoint selection ===" \
      | tee -a "$OUT/run.log"
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u "$CODE/train.py" \
      --data-root "$ROOT/data" \
      --output-dir "$run" \
      --device cuda:0 \
      --architecture UnetPlusPlus \
      --encoder-name tu-hrnet_w48 \
      --encoder-weights imagenet \
      --patch-size 512 \
      --input-scale 2.0 \
      --samples-per-image 12 \
      --batch-size 4 \
      --accumulation-steps 2 \
      --num-workers 4 \
      --epochs 70 \
      --lr 3e-4 \
      --encoder-lr-mult 1.0 \
      --weight-decay 1e-4 \
      --boundary-head-weight 1.0 \
      --seed "$seed" \
      --val-every 0 \
      --resume \
      >> "$run/train.log" 2>&1
  else
    echo "seed $seed already complete at epoch $state_epoch; preserving checkpoint" \
      | tee -a "$OUT/run.log"
  fi
done

echo "DONE: Model377 seed 1 and 2 full-train checkpoints ready; valid labels were not used." \
  | tee -a "$OUT/run.log"
