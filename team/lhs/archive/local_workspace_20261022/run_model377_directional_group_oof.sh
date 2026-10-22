#!/usr/bin/env bash
set -euo pipefail

ROOT=/content/komap_model1
CODE="$ROOT/model377_directional"
CV_DATA="$ROOT/data_model322_group_cv"
OUT="$ROOT/outputs/model377_directional_group_oof"
DRIVE_ROOT="/content/drive/MyDrive/KoMaP/model377_directional_group_oof"
DRIVE_SYNC="$CODE/experiments/model377_drive_checkpoint_sync.py"
PY=python
GPU_ID=0

[[ -d "$CODE" && -d "$CV_DATA" && -f "$DRIVE_SYNC" ]] || {
  echo "missing isolated Model377 code, fixed grouped-CV data, or checkpoint sync helper" >&2
  exit 2
}
[[ -d "$ROOT/outputs/model374_hrnet_group_oof" && -d "$ROOT/outputs/model376_boundary_ce_group_oof" ]] || {
  echo "required Model374/376 train-OOF references are missing" >&2
  exit 2
}
mkdir -p "$OUT" "$DRIVE_ROOT"

link_exact() {
  local link_path=$1 target=$2
  if [[ -L "$link_path" ]]; then
    [[ "$(readlink -f "$link_path")" == "$(readlink -f "$target")" ]] || {
      echo "unexpected symlink target: $link_path -> $(readlink "$link_path")" >&2
      exit 3
    }
  elif [[ -e "$link_path" ]]; then
    echo "refusing to replace existing non-symlink: $link_path" >&2
    exit 3
  else
    ln -s "$target" "$link_path"
  fi
}

for fold in 1 2 3 4; do
  data_root="$ROOT/data_model377_fold$fold"
  run="$OUT/fold$fold"
  oof="$OUT/fold${fold}_oof"
  drive_fold="$DRIVE_ROOT/fold$fold"
  mkdir -p "$data_root" "$run" "$drive_fold/checkpoints"
  link_exact "$data_root/train" "$CV_DATA/train_fold$fold"
  link_exact "$data_root/valid" "$CV_DATA/oof_fold$fold"

  "$PY" -u "$DRIVE_SYNC" restore \
    --run-dir "$run" \
    --drive-dir "$drive_fold/checkpoints" \
    --model-id 377

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
    exit 4
  fi

  if (( state_epoch < 70 )); then
    echo "=== Model377 directional grouped OOF fold $fold: epoch=$state_epoch/70 ===" \
      | tee -a "$OUT/run.log"
    "$PY" -u "$DRIVE_SYNC" watch \
      --run-dir "$run" \
      --drive-dir "$drive_fold/checkpoints" \
      --interval 5 \
      --stable-seconds 8 \
      --keep 2 \
      --model-id 377 \
      >> "$OUT/fold${fold}_drive_sync.log" 2>&1 &
    drive_sync_pid=$!
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u "$CODE/train.py" \
      --data-root "$data_root" \
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
      --boundary-ce-weight 2.0 \
      --boundary-ce-radius 3 \
      --boundary-ce-sigma 1.5 \
      --boundary-ce-warmup-fraction 0.1 \
      --directional-unlikelihood-weight 2.0 \
      --directional-unlikelihood-warmup-fraction 0.1 \
      --seed 0 \
      --val-every 0 \
      --resume \
      2>&1 | tee -a "$run/train.log"
    train_status=$?
    set -e
    kill "$drive_sync_pid" 2>/dev/null || true
    wait "$drive_sync_pid" 2>/dev/null || true
    if (( train_status != 0 )); then
      echo "Model377 fold $fold training exited with status $train_status; resumable snapshots are in Drive." >&2
      exit "$train_status"
    fi
  else
    echo "fold $fold is already complete at epoch $state_epoch; preserving checkpoint" \
      | tee -a "$OUT/run.log"
  fi

  state_epoch=$("$PY" - "$run/train_state.pt" <<'PY'
import sys, torch
state = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(state["epoch"]))
PY
)
  if (( state_epoch != 70 )); then
    echo "expected fold $fold checkpoint at epoch 70, found $state_epoch" >&2
    exit 5
  fi
  if [[ ! -s "$oof/valid_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u "$CODE/evaluate_image_mean.py" \
      --checkpoint "$run/last.pt" \
      --data-root "$data_root" \
      --split valid \
      --output-dir "$oof" \
      --device cuda:0 \
      --batch-size 2 \
      2>&1 | tee -a "$OUT/fold${fold}_oof_eval.log"
  fi

  "$PY" - "$fold" "$run" "$oof" <<'PY'
import hashlib, json, sys
from pathlib import Path
fold = int(sys.argv[1])
run, oof = map(Path, sys.argv[2:])
args = json.loads((run / "args.json").read_text())
assert args["directional_unlikelihood_weight"] == 2.0
assert args["directional_unlikelihood_warmup_fraction"] == 0.1
assert args["boundary_ce_weight"] == 2.0
summary = json.loads((oof / "valid_summary.json").read_text())
checkpoint = run / "last.pt"
manifest = {
    "model": 377,
    "parent_recipe": 376,
    "fold": fold,
    "training_images_in_fold": 52,
    "total_oof_images": 70,
    "heldout_split": f"oof_fold{fold}",
    "epochs": 70,
    "boundary_ce": {"max_weight": 2.0, "radius": 3, "sigma": 1.5,
                    "warmup_fraction": 0.1, "per_image_normalized": True},
    "directional_unlikelihood": {
        "target_gt_class": 3,
        "support": "unique GT Eutectic pixels with an axial 4-neighbor GT Al3Ni",
        "suppressed_class": 1,
        "loss": "mean_over_full_patch(logsumexp(all)-logsumexp(excluding_Al3Ni))",
        "max_beta": 2.0,
        "warmup_fraction": 0.1,
    },
    "metric": summary,
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
}
(oof / "model377_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Model377 fold {fold} OOF summary: {json.dumps(summary, sort_keys=True)}", flush=True)
PY

  "$PY" -u "$DRIVE_SYNC" finalize \
    --run-dir "$run" \
    --oof-dir "$oof" \
    --drive-fold-dir "$drive_fold" \
    --model-id 377
  echo "=== Model377 fold $fold complete ===" | tee -a "$OUT/run.log"
done

"$PY" -u "$CODE/experiments/model377_oof_evaluation.py" \
  --cv-root "$CV_DATA" \
  --model374-root "$ROOT/outputs/model374_hrnet_group_oof" \
  --model376-root "$ROOT/outputs/model376_boundary_ce_group_oof" \
  --model377-root "$OUT" \
  --draws 20000 \
  --output "$OUT/model377_oof_evaluation.json" \
  2>&1 | tee -a "$OUT/evaluation.log"

echo "DONE: Model377 train-only grouped OOF and pre-registered comparisons complete; valid/test were not used." \
  | tee -a "$OUT/run.log"
