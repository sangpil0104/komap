#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-train}"
case "$MODE" in
  train|preflight) ;;
  *) echo "Usage: $0 [preflight|train]" >&2; exit 2 ;;
esac

if [[ -n "${KOMAP_RUN_ROOT:-}" ]]; then
  RUN_ROOT="$KOMAP_RUN_ROOT"
  if [[ -e "$RUN_ROOT" ]]; then
    echo "Refusing to reuse existing run directory: $RUN_ROOT" >&2
    exit 2
  fi
  mkdir -p "$RUN_ROOT"
else
  mkdir -p "$ROOT/runs"
  RUN_ROOT="$(mktemp -d "$ROOT/runs/model349_${MODE}.XXXXXX")"
fi

if [[ -n "${KOMAP_PYTHON:-}" ]]; then
  PY="$KOMAP_PYTHON"
elif [[ -x "$ROOT/.venv-model349/bin/python" ]]; then
  PY="$ROOT/.venv-model349/bin/python"
elif [[ -x /workspace/venv-komap/bin/python ]]; then
  PY=/workspace/venv-komap/bin/python
else
  PY=python3
fi

if [[ -n "${KOMAP_SETUP_ROOT:-}" ]]; then
  cp -a "$ROOT" "$RUN_ROOT/source_snapshot"
  ROOT="$RUN_ROOT/source_snapshot"
  cp "$KOMAP_SETUP_ROOT/provenance.json" "$RUN_ROOT/provenance.json"
  cp "$KOMAP_SETUP_ROOT/source_manifest.json" "$RUN_ROOT/source_manifest.json"
  cp "$KOMAP_SETUP_ROOT/reports/local_patches.diff" "$RUN_ROOT/local_patches.diff"
  cp "$KOMAP_SETUP_ROOT/datasets/manifest.json" "$RUN_ROOT/dataset_manifest.json"
  "$PY" -m pip freeze > "$RUN_ROOT/environment.freeze.txt"
fi

export PYTHONPATH="$ROOT/experiments:$ROOT/experiments/historical_scripts${PYTHONPATH:+:$PYTHONPATH}"

DATA="$RUN_ROOT/data"
OUTPUTS="$RUN_ROOT/outputs"
mkdir -p "$DATA" "$RUN_ROOT/logs"
for archive in train valid; do
  archive_path="${KOMAP_DATA_DIR:-$ROOT/datasets}/${archive}.zip"
  [[ -s "$archive_path" ]] || { echo "Missing external dataset: $archive_path" >&2; exit 2; }
  unzip -oq "$archive_path" -d "$DATA"
done

train_images=$(find "$DATA/train/images" -maxdepth 1 -type f -name '*_image.png' | wc -l | tr -d ' ')
train_masks=$(find "$DATA/train/masks" -maxdepth 1 -type f -name '*_mask.png' | wc -l | tr -d ' ')
valid_images=$(find "$DATA/valid/images" -maxdepth 1 -type f -name '*_image.png' | wc -l | tr -d ' ')
valid_masks=$(find "$DATA/valid/masks" -maxdepth 1 -type f -name '*_mask.png' | wc -l | tr -d ' ')
if [[ "$train_images" != 70 || "$train_masks" != 70 || "$valid_images" != 20 || "$valid_masks" != 20 ]]; then
  echo "Expected train=70 image/mask pairs and valid=20; got train=$train_images/$train_masks valid=$valid_images/$valid_masks" >&2
  exit 2
fi

ASSETS="$ROOT/bundle_assets/training"
[[ -s "$ASSETS/model303_progress.json" ]] || {
  echo "Missing grouped-CV manifest: $ASSETS/model303_progress.json" >&2; exit 2;
}
if [[ "$MODE" == preflight ]]; then
  "$PY" "$ROOT/experiments/model322_team_b4/KoMaP/prepare_group_cv.py" \
    --data "$DATA" --manifest "$ASSETS/model303_progress.json" \
    --out "$RUN_ROOT/data_model322_group_cv"
  echo "PREFLIGHT OK: 70 train pairs, 20 valid pairs, and four specimen-group folds"
  exit 0
fi

"$PY" -c 'import numpy, PIL, torch; print("python", __import__("sys").version.split()[0], "torch", torch.__version__, "numpy", numpy.__version__)'
echo "mode=$MODE run_root=$RUN_ROOT"
echo "official metric: image-wise four-class mIoU, empty union = 0"

H="$ROOT/experiments/historical_scripts"
run_stage() {
  local name="$1"; shift
  echo "Running $name"
  "$PY" -u "$@" 2>&1 | tee "$RUN_ROOT/logs/${name}.log"
}

# Generate every ResNet checkpoint in the historical Model349 ancestry.
run_stage train_model6 "$H/train_models_2_to_6.py" --model-id 6 \
  --data-dir "$DATA" --output-dir "$OUTPUTS/model6_unetpp_resnet34" \
  --epochs 30 --image-size 512 --num-workers 4 --seed 42
run_stage train_model62 "$H/improve_model6.py" --data-dir "$DATA" \
  --checkpoint "$OUTPUTS/model6_unetpp_resnet34/best.pt" \
  --output-dir "$OUTPUTS/model62_ft_weighted_safeaug" --epochs 20 \
  --batch-size 4 --encoder-lr 5e-6 --decoder-lr 5e-5 \
  --loss-mode weighted --aug-mode safe_plus --select-tta --seed 42
run_stage train_model97 "$H/improve_five_causes.py" --data-dir "$DATA" \
  --checkpoint "$OUTPUTS/model62_ft_weighted_safeaug/best.pt" \
  --output-dir "$OUTPUTS/model97_native_patch" --mode train \
  --patch-size 512 --patches-per-image 6 --val-stride 448 \
  --final-stride 384 --freeze-epochs 3 --finetune-epochs 12 \
  --encoder-lr 5e-6 --decoder-lr 1e-4 --seed 42
run_stage train_model98 "$H/hard_negative_finetune.py" --data-dir "$DATA" \
  --checkpoint "$OUTPUTS/model97_native_patch/best.pt" \
  --output-dir "$OUTPUTS/model98_hard_negative" --epochs 4 \
  --encoder-lr 1e-6 --decoder-lr 5e-5 --seed 42
run_stage train_model109 "$H/encoder_structure_search.py" --candidate aspp \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model98_hard_negative/best.pt" \
  --output-dir "$OUTPUTS/model109_aspp_native" --epochs 4 \
  --patch-size 512 --val-stride 384 --patches-per-image 6 \
  --lr 1.5e-5 --eval-bias -0.2 --seed 42 --skip-test-inference

# Historical Model183 base: three route47 probabilities plus five independently
# trained source checkpoints and a representation branch.
run_stage train_restart51 "$H/encoder_structure_search.py" --candidate aspp \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model109_aspp_native/best.pt" \
  --output-dir "$OUTPUTS/restart51_aspp_768_finetune" \
  --patch-size 768 --val-stride 576 --tile-batch-size 1 \
  --patches-per-image 4 --epochs 6 --batch-size 2 --lr 8e-6 \
  --eval-bias -0.55 --seed 123 --skip-test-inference
run_stage train_restart59 "$H/encoder_structure_search.py" --candidate aspp \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model109_aspp_native/best.pt" \
  --output-dir "$OUTPUTS/restart59_aspp768_seed456" \
  --patch-size 768 --val-stride 576 --tile-batch-size 1 \
  --patches-per-image 4 --epochs 5 --batch-size 2 --lr 8e-6 \
  --eval-bias -0.55 --seed 456 --skip-test-inference
run_stage train_restart61 "$H/pairwise_margin_finetune.py" --data-dir "$DATA" \
  --checkpoint "$OUTPUTS/restart51_aspp_768_finetune/best.pt" \
  --output-dir "$OUTPUTS/restart61_pairwise_margin" --batch-size 1 \
  --num-workers 2 --seed 42
run_stage train_model123 "$H/resnet34_refinements.py" --candidate ema \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model109_aspp_native/best.pt" \
  --output-dir "$OUTPUTS/model123_ema" --patch-size 512 \
  --val-stride 384 --patches-per-image 6 --epochs 4 --batch-size 4 \
  --lr 1e-5 --eval-bias -0.35 --sdf-weight 0.2 \
  --warmup-epochs 2 --ema-decay 0.995 --seed 42
run_stage train_model181 "$H/model181_raw_zcdf_aspp.py" --data-dir "$DATA" \
  --checkpoint "$OUTPUTS/model109_aspp_native/best.pt" \
  --output-dir "$OUTPUTS/model181_raw_zcdf_aspp" \
  --patch-size 512 --patches-per-image 8 --batch-size 2 --epochs 5 \
  --encoder-lr 1e-5 --decoder-lr 3e-5 \
  --selection-metric image_mean_miou --seed 181

run_stage infer_route47 "$H/route_upper_bound.py" --data-dir "$DATA" \
  --old-checkpoint "$OUTPUTS/model97_native_patch/best.pt" \
  --new-checkpoint "$OUTPUTS/model98_hard_negative/best.pt" \
  --aspp-base-checkpoint "$OUTPUTS/model98_hard_negative/best.pt" \
  --aspp-checkpoint "$OUTPUTS/model109_aspp_native/best.pt" \
  --output-dir "$OUTPUTS/restart47_router" \
  --output-json "$OUTPUTS/restart47_router/results.json"
run_stage build_model183_cache "$ROOT/experiments/build_model183_entries.py" \
  --data-dir "$DATA" --cache-dir "$OUTPUTS/restart47_router/cache" \
  --representation-checkpoint "$OUTPUTS/model181_raw_zcdf_aspp/best.pt" \
  --pair-checkpoint "$OUTPUTS/restart61_pairwise_margin/best.pt" \
  --source-restart51 "$OUTPUTS/restart51_aspp_768_finetune/best.pt" \
  --source-model109 "$OUTPUTS/model109_aspp_native/best.pt" \
  --source-seed456 "$OUTPUTS/restart59_aspp768_seed456/best.pt" \
  --source-pairwise "$OUTPUTS/restart61_pairwise_margin/best.pt" \
  --source-ema "$OUTPUTS/model123_ema/best.pt" \
  --output-json "$OUTPUTS/model183_metric_calibration/results.json"

run_stage train_model194 "$H/model194_multistage_team_schedule.py" \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model109_aspp_native/best.pt" \
  --output-dir "$OUTPUTS/model194_multistage_seed2028" \
  --stage0-epochs 3 --stage1-epochs 3 --stage2-epochs 0 \
  --stage0-patches 16 --stage1-patches 16 --stage2-patches 8 \
  --stage0-batch 2 --stage1-batch 2 --stage2-batch 1 \
  --encoder-lr 1e-4 --decoder-lr 3e-4 \
  --selection-metric image_mean_miou --seed 2028

CV_DATA="$RUN_ROOT/data_model322_group_cv"
"$PY" "$ROOT/experiments/model322_team_b4/KoMaP/prepare_group_cv.py" \
  --data "$DATA" --manifest "$ASSETS/model303_progress.json" --out "$CV_DATA" \
  2>&1 | tee "$RUN_ROOT/logs/prepare_group_cv.log"

echo "Training Model195 native ResNet34-ASPP continuation."
"$PY" -u "$ROOT/experiments/historical_scripts/model194_multistage_team_schedule.py" \
  --data-dir "$DATA" \
  --checkpoint "$OUTPUTS/model194_multistage_seed2028/best.pt" \
  --output-dir "$OUTPUTS/model195_native_resume" \
  --stage0-epochs 0 --stage1-epochs 0 --stage2-epochs 2 \
  --stage2-patches 8 --stage2-batch 2 \
  --eval-patch-size 768 --eval-stride 576 --tile-batch-size 2 \
  --encoder-lr 5e-5 --decoder-lr 1.5e-4 --weight-decay 1e-4 \
  --selection-metric image_mean_miou --seed 2028 --num-workers 4 \
  2>&1 | tee "$RUN_ROOT/logs/train_model195.log"

echo "Generating the recorded Model220 20-step TENT probability branch."
"$PY" -u "$ROOT/experiments/historical_scripts/model217_native_tent.py" \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model195_native_resume/best.pt" \
  --output-dir "$OUTPUTS/model220_native_tent_extended" \
  --patch-size 768 --stride 576 --tile-batch-size 2 \
  --adapt 10 1e-5 --adapt 10 2e-5 --adapt 10 5e-5 \
  --adapt 20 1e-5 --adapt 20 2e-5 --adapt 20 5e-5 \
  2>&1 | tee "$RUN_ROOT/logs/infer_model220_tent.log"

run_stage train_model239 "$H/model239_native_stem_finetune.py" \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model195_native_resume/best.pt" \
  --output-dir "$OUTPUTS/model239_native_stem" \
  --epochs 4 --encoder-lr 1e-5 --decoder-lr 3e-5 \
  --native-lr 3e-4 --seed 239
run_stage train_model242 "$H/model242_ema_continuation.py" \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model239_native_stem/best.pt" \
  --output-dir "$OUTPUTS/model242_native_stem_ema" \
  --epochs 4 --ema-decay 0.99 --encoder-lr 5e-6 \
  --decoder-lr 1.5e-5 --native-lr 1e-4 --seed 242
run_stage train_model257 "$H/model253_highres_scale_finetune.py" \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model242_native_stem_ema/best_ema.pt" \
  --output-dir "$OUTPUTS/model257_full_image_stats_1x" \
  --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 \
  --epochs 4 --scheduler-epochs 4 --encoder-lr 5e-6 \
  --decoder-lr 1.5e-5 --native-lr 1e-4 --weight-decay 1e-4 \
  --val-stride 576 --tta d4 --selection-metric image_mean_miou \
  --seed 257 --global-representation --num-workers 4

echo "Training Model269 full-stat ResNet34-ASPP continuation."
"$PY" -u "$ROOT/experiments/historical_scripts/model253_highres_scale_finetune.py" \
  --data-dir "$DATA" --checkpoint "$OUTPUTS/model257_full_image_stats_1x/best.pt" \
  --output-dir "$OUTPUTS/model269_fullstats_1x_extended" \
  --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 \
  --epochs 4 --scheduler-epochs 4 --encoder-lr 2.5e-6 \
  --decoder-lr 7.5e-6 --native-lr 5e-5 --weight-decay 1e-4 \
  --val-stride 576 --tta d4 --selection-metric image_mean_miou \
  --seed 269 --global-representation --num-workers 4 \
  2>&1 | tee "$RUN_ROOT/logs/train_model269.log"

train_b4_family() {
  local model="$1" code="$2" out="$3" norm="$4" input="$5"
  local scale="$6" target_width="$7" sample="$8"
  local fold run
  mkdir -p "$out"
  for fold in 1 2 3 4; do
    run="$out/fold$fold"
    mkdir -p "$run"
    echo "Training Model${model} fold ${fold}/4 (16000 updates)."
    "$PY" -u "$code/train.py" --data "$CV_DATA" \
      --train_splits "train_fold$fold" --out "$run" \
      --arch unet --encoder efficientnet-b4 --norm "$norm" --input "$input" \
      --crop 1536 --scale "$scale" --target_width "$target_width" \
      --sample "$sample" --bs 2 --iters 16000 --lr 3e-4 --wd 1e-2 \
      --dice_w 1 --cldice_w 0 --seed 0 --workers 4 --no_valid_eval \
      2>&1 | tee "$run/train.log"
    "$PY" "$code/predict.py" --data "$CV_DATA" \
      --split "oof_fold$fold" --runs "$run" \
      --out "$out/fold${fold}_oof" --save_probs \
      2>&1 | tee "$out/fold${fold}_oof.log"
  done
  "$PY" "$code/aggregate_oof.py" --runs "$out" --folds 4 --expected-images 70 \
    2>&1 | tee "$out/aggregate_oof.log"
  "$PY" "$code/predict.py" --data "$DATA" --split valid \
    --runs "$out/fold1" "$out/fold2" "$out/fold3" "$out/fold4" \
    --out "$out/valid_ensemble" --save_probs \
    2>&1 | tee "$out/valid_ensemble.log"
  echo "Model${model} OOF training and valid probability inference complete."
}

echo "Training grouped EfficientNet-B4 component branches (12 folds total)."
train_b4_family 322 \
  "$ROOT/experiments/model322_team_b4/KoMaP" \
  "$OUTPUTS/model322_team_b4_group_oof" imagenet gray3 3 0 area
train_b4_family 326 \
  "$ROOT/experiments/model326_hybrid_fullstats/KoMaP" \
  "$OUTPUTS/model326_hybrid_fullstats_group_oof" image_global zcdf3 3 0 area
train_b4_family 329 \
  "$ROOT/experiments/model326_hybrid_fullstats/KoMaP" \
  "$OUTPUTS/model329_fullstats_width4000_uniform_group_oof" image_global zcdf3 1 4000 uniform

echo "Training Model336 HRNet-W48/U-Net++ and its Model347 weighted-CE fine-tune."
MODEL336_OUT="$OUTPUTS/model336_hrnet_best"
"$PY" -u "$ROOT/experiments/model336_hrnet_best/train.py" \
  --data-root "$DATA" --output-dir "$MODEL336_OUT" \
  --batch-size 4 --accumulation-steps 2 --epochs 70 --samples-per-image 12 \
  --lr 3e-4 --encoder-lr-mult 1.0 --weight-decay 1e-4 \
  --boundary-head-weight 1.0 --seed 0 --val-every 5 \
  2>&1 | tee "$RUN_ROOT/logs/train_model336.log"
"$PY" -u "$ROOT/experiments/model336_hrnet_best/evaluate_image_mean.py" \
  --checkpoint "$MODEL336_OUT/last.pt" --data-root "$DATA" --split valid \
  --output-dir "$MODEL336_OUT/valid_eval" --device cuda:0 --batch-size 2 \
  2>&1 | tee "$RUN_ROOT/logs/eval_model336.log"

MODEL347_OUT="$OUTPUTS/model347_hrnet_inverse_ce_finetune"
"$PY" -u "$ROOT/experiments/model336_hrnet_best/train.py" \
  --data-root "$DATA" --output-dir "$MODEL347_OUT" \
  --batch-size 4 --accumulation-steps 2 --epochs 12 --samples-per-image 12 \
  --lr 1e-4 --encoder-lr-mult 0.3 --weight-decay 1e-4 \
  --boundary-head-weight 1.0 --seed 347 --val-every 0 \
  --init-checkpoint "$MODEL336_OUT/last.pt" \
  --ce-class-weights 0.22906544 0.49862612 1.39166570 1.88064275 \
  2>&1 | tee "$RUN_ROOT/logs/train_model347.log"
"$PY" -u "$ROOT/experiments/model336_hrnet_best/evaluate_image_mean.py" \
  --checkpoint "$MODEL347_OUT/last.pt" --data-root "$DATA" --split valid \
  --output-dir "$MODEL347_OUT/valid_eval" --device cuda:0 --batch-size 2 \
  2>&1 | tee "$RUN_ROOT/logs/eval_model347.log"

echo "Rebuilding the selected route chain from the newly generated branch probabilities."
ROUTE="$ROOT/experiments/historical_scripts/model335_exhaustive_b4_route_search.py"
"$PY" -u "$ROUTE" \
  --entries "$OUTPUTS/model183_metric_calibration/entries.pkl" \
  --model195 "$OUTPUTS/model195_native_resume/valid_probs.npz" \
  --tent20 "$OUTPUTS/model220_native_tent_extended/valid_probs_06.npz" \
  --model269 "$OUTPUTS/model269_fullstats_1x_extended/valid_probs.npz" \
  --model322 "$OUTPUTS/model322_team_b4_group_oof/valid_ensemble/probabilities.npz" \
  --model326 "$OUTPUTS/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz" \
  --workers "${KOMAP_WORKERS:-4}" \
  --output "$OUTPUTS/model335_exhaustive_b4_routes/results.json" \
  2>&1 | tee "$RUN_ROOT/logs/route_model335.log"
"$PY" "$ROOT/experiments/historical_scripts/model338_fine_gamma_route_search.py" \
  --root "$RUN_ROOT" --workers "${KOMAP_WORKERS:-4}" \
  --output "$OUTPUTS/model338_fine_gamma_route_search/results.json" \
  2>&1 | tee "$RUN_ROOT/logs/route_model338.log"
"$PY" "$ROOT/experiments/model339_joint_gamma_route_search.py" \
  --root "$RUN_ROOT" --workers "${KOMAP_WORKERS:-4}" \
  --output "$OUTPUTS/model339_joint_gamma_route_search/results.json" \
  2>&1 | tee "$RUN_ROOT/logs/route_model339.log"
"$PY" "$ROOT/experiments/model341_model329_joint_route_search.py" \
  --root "$RUN_ROOT" --workers "${KOMAP_WORKERS:-4}" \
  --output "$OUTPUTS/model341_model329_joint_route_search/results.json" \
  2>&1 | tee "$RUN_ROOT/logs/route_model341.log"
"$PY" "$ROOT/experiments/model346_model336_into_341_route.py" \
  --root "$RUN_ROOT" --workers "${KOMAP_WORKERS:-4}" --bootstrap 20000 \
  --output "$OUTPUTS/model346_model336_into_341_route/results.json" \
  2>&1 | tee "$RUN_ROOT/logs/route_model346.log"
"$PY" "$ROOT/experiments/model346_model336_into_341_route.py" \
  --root "$RUN_ROOT" --expert-id 347 --anchor-model346 \
  --workers "${KOMAP_WORKERS:-4}" --bootstrap 20000 \
  --output "$OUTPUTS/model349_model347_into_346_route/results.json" \
  2>&1 | tee "$RUN_ROOT/logs/route_model349.log"

"$PY" - "$OUTPUTS/model349_model347_into_346_route/results.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"TRAINED PIPELINE Model349 valid image-wise mIoU (empty=0): "
      f"{d['raw_best_valid_image_mean_miou_empty0']:.10%}")
print(f"selected Model347 gamma: {d['raw_best_gamma347']}")
PY
echo "Completed full Model349 training/retraining pipeline: $RUN_ROOT"
