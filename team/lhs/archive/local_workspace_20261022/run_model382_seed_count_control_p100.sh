#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/komap_model1
CODE="$ROOT/model336_hrnet_best"
CV_DATA="$ROOT/data_model322_group_cv"
OUT="$ROOT/outputs/model382_seed_count_control"
PY=/workspace/venv-komap/bin/python
GPU_ID=7

[[ -d "$CODE" && -d "$CV_DATA" && -x "$PY" ]] || {
  echo "missing Model374 code, fixed grouped-CV data, or KoMaP Python environment" >&2
  exit 2
}
for fold in 1 2 3 4; do
  [[ -d "$CV_DATA/train_fold$fold" && -d "$CV_DATA/oof_fold$fold" ]] || {
    echo "missing fixed train/OOF split for fold $fold" >&2
    exit 2
  }
done

check_gpu_idle() {
  local gpu_row gpu_mem gpu_util
  gpu_row=$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  gpu_row=${gpu_row// /}
  IFS=, read -r gpu_mem gpu_util <<< "$gpu_row"
  if (( gpu_mem > 512 || gpu_util > 0 )); then
    echo "refusing to use GPU $GPU_ID: memory=${gpu_mem}MiB utilization=${gpu_util}%" >&2
    exit 3
  fi
}
check_gpu_idle
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'

if [[ ! -e "$OUT" ]]; then
  mkdir "$OUT"
fi
if [[ -e "$OUT/protocol.json" ]]; then
  "$PY" - "$OUT/protocol.json" "$0" <<'PY'
import hashlib, json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text())
runner = Path(sys.argv[2])
digest = hashlib.sha256(runner.read_bytes()).hexdigest()
if manifest.get("model") != 382 or manifest.get("runner_sha256") != digest:
    raise SystemExit("existing Model382 output belongs to another protocol/runner; refusing resume")
PY
else
  if find "$OUT" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "Model382 output directory is non-empty but has no protocol manifest; refusing to adopt it" >&2
    exit 4
  fi
  "$PY" - "$OUT/protocol.json" "$0" <<'PY'
import hashlib, json, sys
from pathlib import Path
output, runner = Path(sys.argv[1]), Path(sys.argv[2])
record = {
    "model": 382,
    "recipe": "Model374 HRNet-W48 U-Net++ grouped OOF; seeds 1 and 2",
    "seeds": [1, 2],
    "folds": [1, 2, 3, 4],
    "epochs": 70,
    "gpu_index": 7,
    "official_valid_or_test_used": False,
    "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
}
output.write_text(json.dumps(record, indent=2) + "\n")
PY
fi

if ! mkdir "$OUT/.model382.lock" 2>/dev/null; then
  echo "Model382 runner lock exists; refusing a concurrent launch" >&2
  exit 5
fi
trap 'rmdir "$OUT/.model382.lock" 2>/dev/null || true' EXIT

safe_link() {
  local link_path=$1 target=$2
  if [[ -L "$link_path" ]]; then
    [[ "$(readlink -f "$link_path")" == "$(readlink -f "$target")" ]] || {
      echo "unexpected symlink target: $link_path -> $(readlink "$link_path")" >&2
      exit 6
    }
  elif [[ -e "$link_path" ]]; then
    echo "refusing to replace existing path: $link_path" >&2
    exit 6
  else
    ln -s "$target" "$link_path"
  fi
}

for seed in 1 2; do
  for fold in 1 2 3 4; do
    data_root="$ROOT/data_model382_seed${seed}_fold${fold}"
    run="$OUT/seed${seed}/fold${fold}"
    oof="$OUT/seed${seed}/fold${fold}_oof"
    mkdir -p "$data_root" "$run" "$oof"
    safe_link "$data_root/train" "$CV_DATA/train_fold$fold"
    safe_link "$data_root/valid" "$CV_DATA/oof_fold$fold"

    if [[ -s "$run/args.json" ]]; then
      "$PY" - "$run/args.json" "$data_root" "$run" "$seed" <<'PY'
import json, sys
from pathlib import Path
args = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "data_root": sys.argv[2], "output_dir": sys.argv[3],
    "architecture": "UnetPlusPlus", "encoder_name": "tu-hrnet_w48",
    "encoder_weights": "imagenet", "patch_size": 512, "input_scale": 2.0,
    "samples_per_image": 12, "batch_size": 4, "accumulation_steps": 2,
    "num_workers": 4, "epochs": 70, "lr": 3e-4,
    "encoder_lr_mult": 1.0, "weight_decay": 1e-4,
    "boundary_head_weight": 1.0, "seed": int(sys.argv[4]),
    "val_every": 0, "init_checkpoint": None,
}
for key, value in expected.items():
    if args.get(key) != value:
        raise SystemExit(f"resume config mismatch for {key}: {args.get(key)!r} != {value!r}")
PY
    elif [[ -e "$run/last.pt" || -e "$run/train_state.pt" ]]; then
      echo "checkpoint exists without args.json: $run" >&2
      exit 7
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
      exit 7
    fi

    if (( state_epoch < 70 )); then
      check_gpu_idle
      printf '\n[%s] Model382 seed=%s fold=%s training from epoch=%s/70 on P100 GPU %s\n' \
        "$(date -Is)" "$seed" "$fold" "$state_epoch" "$GPU_ID" | tee -a "$OUT/run.log"
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
        --seed "$seed" \
        --val-every 0 \
        --resume \
        2>&1 | tee -a "$run/train.log"
    else
      echo "[$(date -Is)] Model382 seed=$seed fold=$fold already at epoch 70; preserving checkpoint" \
        | tee -a "$OUT/run.log"
    fi

    if [[ ! -s "$oof/valid_summary.json" ]]; then
      if [[ -e "$oof/valid_probabilities.npz" ]]; then
        echo "partial OOF probability archive exists without summary; refusing overwrite: $oof" >&2
        exit 8
      fi
      printf '[%s] Model382 seed=%s fold=%s OOF inference\n' "$(date -Is)" "$seed" "$fold" \
        | tee -a "$OUT/run.log"
      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u "$CODE/evaluate_image_mean.py" \
        --checkpoint "$run/last.pt" \
        --data-root "$data_root" \
        --split valid \
        --output-dir "$oof" \
        --device cuda:0 \
        --batch-size 2 \
        2>&1 | tee -a "$OUT/seed${seed}_fold${fold}_oof_eval.log"
    fi

    "$PY" - "$seed" "$fold" "$run" "$oof" "$CV_DATA" "$CODE" <<'PY'
import hashlib, json, sys
from pathlib import Path
import torch
seed, fold = map(int, sys.argv[1:3])
run, oof, cv_data, code = map(Path, sys.argv[3:])
args = json.loads((run / "args.json").read_text())
state = torch.load(run / "train_state.pt", map_location="cpu", weights_only=False)
if int(state.get("epoch", -1)) != 70:
    raise SystemExit(f"incomplete training: seed={seed}, fold={fold}, epoch={state.get('epoch')}")
train_log = (run / "train.log").read_text(encoding="utf-8", errors="replace")
if "done. final model:" not in train_log or "Traceback" in train_log:
    raise SystemExit(f"training log does not show clean completion: {run}")
data_root = Path(args["data_root"])
if (data_root / "train").resolve() != (cv_data / f"train_fold{fold}").resolve():
    raise SystemExit("training data is not the pre-registered train fold")
if (data_root / "valid").resolve() != (cv_data / f"oof_fold{fold}").resolve():
    raise SystemExit("inference data is not the pre-registered held-out fold")
if (oof / "valid_probabilities.npz").stat().st_size == 0:
    raise SystemExit("empty OOF probability archive")
summary = json.loads((oof / "valid_summary.json").read_text())
if summary.get("split") != "valid" or summary.get("image_miou_empty_zero") is None:
    raise SystemExit("unexpected OOF summary or empty=0 metric missing")
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
record = {
    "model": 382, "seed": seed, "fold": fold,
    "train_images": str((cv_data / f"train_fold{fold}").resolve()),
    "heldout_images": str((cv_data / f"oof_fold{fold}").resolve()),
    "epochs": int(state["epoch"]), "metric": "image-wise 4-class mIoU; empty-union=0",
    "oof_image_miou_empty_zero": float(summary["image_miou_empty_zero"]),
    "checkpoint_sha256": sha256(run / "last.pt"),
    "oof_probabilities_sha256": sha256(oof / "valid_probabilities.npz"),
    "training_args": args,
}
(oof / "model382_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({"seed": seed, "fold": fold, "oof_image_miou_empty_zero": record["oof_image_miou_empty_zero"],
                  "checkpoint_sha256": record["checkpoint_sha256"],
                  "oof_sha256": record["oof_probabilities_sha256"]}, indent=2))
PY
  done
done

"$PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
records = []
for seed in (1, 2):
    for fold in (1, 2, 3, 4):
        path = root / f"seed{seed}" / f"fold{fold}_oof" / "model382_manifest.json"
        records.append(json.loads(path.read_text()))
if len(records) != 8:
    raise SystemExit("expected all 8 seed/fold manifests")
(root / "all_seed_runs_complete.json").write_text(json.dumps({"model": 382, "completed_runs": records}, indent=2) + "\n")
PY
echo "[$(date -Is)] Model382 training and per-fold OOF complete; ensemble scoring remains gated on baseline OOF integrity." \
  | tee -a "$OUT/run.log"
