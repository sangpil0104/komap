#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
PY=/workspace/venv-komap/bin/python
CODE="$ROOT/model336_hrnet_best"
CV_DATA="$ROOT/data_model322_group_cv"
OUT="$ROOT/outputs/model374_hrnet_group_oof"
GPU_ID=6
FOLDS="${FOLDS:-1 2 3 4}"

[[ -x "$PY" ]] || { echo "missing Python environment: $PY" >&2; exit 2; }
[[ -d "$CODE" && -d "$CV_DATA" ]] || { echo "missing Model336 code or fixed CV data" >&2; exit 2; }
mkdir -p "$OUT"

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

for fold in $FOLDS; do
  case "$fold" in 1|2|3|4) ;; *) echo "invalid fold: $fold" >&2; exit 2 ;; esac
  if [[ "$fold" != "1" ]]; then
    "$PY" - "$OUT/fold1_kill_switch_corrected.json" <<'PY'
import json, sys
from pathlib import Path
gate = Path(sys.argv[1])
if not gate.is_file():
    raise SystemExit(f"official fold-1 gate missing: {gate}")
result = json.loads(gate.read_text())
if not result.get("continue", False):
    raise SystemExit("official Model374 fold-1 kill switch failed; not launching later folds")
print("official Model374 fold-1 kill switch passed")
PY
  fi
  data_root="$ROOT/data_model374_fold$fold"
  run="$OUT/fold$fold"
  oof="$OUT/fold${fold}_oof"
  mkdir -p "$data_root" "$run"
  link_exact "$data_root/train" "$CV_DATA/train_fold$fold"
  link_exact "$data_root/valid" "$CV_DATA/oof_fold$fold"

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
    echo "=== Model374 HRNet grouped OOF fold $fold: fixed 70 epochs; no validation checkpoint selection ===" \
      | tee -a "$OUT/run.log"
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
      --seed 0 \
      --val-every 0 \
      --resume \
      >> "$run/train.log" 2>&1
  else
    echo "fold $fold is already complete at epoch $state_epoch; preserving checkpoint" \
      | tee -a "$OUT/run.log"
  fi

  if [[ ! -s "$oof/valid_summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u "$CODE/evaluate_image_mean.py" \
      --checkpoint "$run/last.pt" \
      --data-root "$data_root" \
      --split valid \
      --output-dir "$oof" \
      --device cuda:0 \
      --batch-size 2 \
      >> "$OUT/fold${fold}_oof_eval.log" 2>&1
  else
    echo "fold $fold OOF evaluation exists; preserve and reuse it: $oof/valid_summary.json" \
      | tee -a "$OUT/run.log"
  fi

  manifest="$oof/inference_manifest.json"
  if [[ ! -s "$manifest" ]]; then
    mkdir -p "$oof"
    "$PY" - "$fold" "$run/args.json" "$run/last.pt" "$data_root" \
      "$CV_DATA" "$manifest" "$CODE" "$0" <<'PY'
import hashlib, json, sys
import torch
from pathlib import Path
fold = int(sys.argv[1])
args_path, checkpoint, data_root, cv_data, output, code_root, runner = map(Path, sys.argv[2:])
args = json.loads(args_path.read_text())
expected_train = (cv_data / f"train_fold{fold}").resolve()
expected_valid = (cv_data / f"oof_fold{fold}").resolve()
if (data_root / "train").resolve() != expected_train:
    raise SystemExit("training symlink does not match the held-out fold manifest")
if (data_root / "valid").resolve() != expected_valid:
    raise SystemExit("OOF symlink does not match the held-out fold manifest")
expected_args = {
    "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
    "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
    "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
    "epochs": 70, "lr": 3e-4, "encoder_lr_mult": 1.0,
    "weight_decay": 1e-4, "boundary_head_weight": 1.0,
    "seed": 0, "val_every": 0, "init_checkpoint": None,
}
for key, expected_value in expected_args.items():
    if args.get(key) != expected_value:
        raise SystemExit(f"unexpected training argument {key}: {args.get(key)!r}")
state = torch.load(checkpoint.parent / "train_state.pt", map_location="cpu", weights_only=False)
if int(state.get("epoch", -1)) != 70:
    raise SystemExit(f"training state is incomplete: epoch={state.get('epoch')}")
train_log = (checkpoint.parent / "train.log").read_text(encoding="utf-8", errors="replace")
if "done. final model:" not in train_log or "Traceback" in train_log:
    raise SystemExit("training log lacks a clean 70-epoch completion marker")
if int(args.get("epochs", -1)) != 70 or args.get("val_every") != 0:
    raise SystemExit("training configuration violates fixed Model374 protocol")
sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
tree_digest = hashlib.sha256()
for source in sorted(code_root.rglob("*.py")):
    tree_digest.update(source.relative_to(code_root).as_posix().encode())
    tree_digest.update(source.read_bytes())
record = {"fold": fold, "training_args": str(args_path), "training_data_root": args["data_root"],
          "checkpoint": str(checkpoint), "checkpoint_sha256": sha,
          "train_state_epoch": int(state["epoch"]),
          "training_args_sha256": sha256(args_path),
          "model_code_tree_sha256": tree_digest.hexdigest(),
          "evaluation_script_sha256": sha256(code_root / "evaluate_image_mean.py"),
          "runner_sha256": sha256(runner),
          "inference_data_root": str(data_root), "inference_split": "valid",
          "heldout_data_root": str(expected_valid)}
output.write_text(json.dumps(record, indent=2) + "\n")
PY
  fi

  if [[ "$fold" == "1" ]]; then
    "$PY" - "$oof/valid_summary.json" "$CV_DATA" \
      "$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz" \
      "$ROOT/model326_hybrid_fullstats/KoMaP" \
      "$OUT/fold1_kill_switch_corrected.json" <<'PY'
import json, sys
import hashlib
from pathlib import Path
import numpy as np
candidate = json.loads(Path(sys.argv[1]).read_text())
cv_data, baseline_path, code = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
baseline_sha = sha256_file(baseline_path)
expected_baseline_sha = "539bc1983bffb3ff55303f6865fc9a2094edc5dea9910551a481005fc1a01837"
if baseline_sha != expected_baseline_sha:
    raise SystemExit(f"Model329 OOF reference archive changed: {baseline_sha}")
sys.path.insert(0, str(code))
from common import list_split, load_mask
def probability_for(archive, name):
    for key in (name, f"{name}_image"):
        if key in archive:
            return np.asarray(archive[key], dtype=np.float32)
    raise KeyError(name)
def model329_to_canonical(probability):
    """Model329 B4 OOF archive is already in canonical/team order."""
    if probability.ndim != 3 or probability.shape[0] != 4:
        raise ValueError(f"invalid raw Model329 probabilities: {probability.shape}")
    return probability
def class_iou(prediction, target):
    matrix = np.bincount(target.astype(np.int64).ravel() * 4 +
                         prediction.astype(np.int64).ravel(), minlength=16).reshape(4, 4)
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return np.divide(intersection, union, out=np.zeros(4), where=union > 0)
score = float(candidate["image_miou_empty_zero"])
reference_scores = []
fold1 = list_split(str(cv_data), "oof_fold1")
with np.load(baseline_path) as archive:
    for image_path, mask_path in fold1:
        name = Path(image_path).name.removesuffix("_image.png")
        prob = model329_to_canonical(probability_for(archive, name))
        reference_scores.append(float(class_iou(prob.argmax(axis=0), load_mask(mask_path)).mean()))
reference = float(np.mean(reference_scores))
delta_pp = (score - reference) * 100.0
passed = delta_pp > -2.0
result = {"candidate_model": 374, "fold": 1, "candidate_oof_image_mean_miou_empty0": score,
          "model329_same_fold_oof_image_mean_miou_empty0": reference,
          "model329_recomputed_from_grouped_train_oof_probabilities": True,
          "model329_archive_class_order": ["Al", "Al3Ni", "Primary_Si", "Eutectic_Si"],
          "model329_to_canonical_channel_indices": [0, 1, 2, 3],
          "model329_oof_reference_sha256": baseline_sha,
          "reference_images": len(reference_scores),
          "delta_pp": delta_pp, "kill_switch_threshold_pp": -2.0, "continue": passed}
Path(sys.argv[5]).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
if not passed:
    raise SystemExit("Model374 fold-1 kill switch failed: >2pp below Model329 OOF")
PY
    "$PY" -u "$ROOT/experiments/model374_fold1_futility_gate.py" \
      --cv-data "$CV_DATA" \
      --candidate "$oof/valid_probabilities.npz" \
      --reference "$ROOT/outputs/model329_fullstats_width4000_uniform_group_oof/train_oof_probs.npz" \
      --common-dir "$ROOT/model326_hybrid_fullstats/KoMaP" \
      --output "$OUT/fold1_futility_gate_corrected.json"
  fi
  echo "=== Model374 fold $fold complete ===" | tee -a "$OUT/run.log"
done

echo "DONE: requested Model374 folds complete; no official validation labels were opened." \
  | tee -a "$OUT/run.log"
