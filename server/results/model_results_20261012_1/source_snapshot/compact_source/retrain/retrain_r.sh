#!/usr/bin/env bash
# Retrain the 14 R (exh14) weights for one seed with the recorded Model349 recipes.
#   bash retrain/retrain_r.sh SEED TRAIN_ROUTE_ROOT DATA_ROOT OUT [GPU]
# TRAIN_ROUTE_ROOT : unzipped Model349_train_route_code.zip (contains experiments/ and bundle_assets/)
# DATA_ROOT        : folder with train/{images,masks} (70) and valid/{images,masks} (20)
# OUT              : new folder; weights are written to OUT/weights/<name>/ with the exh14 checkpoint names
# Every recorded seed N becomes N + 1000*SEED (SEED=0 = the recorded seeds). Commands are copied from
# experiments/run_model349_pipeline.sh. Only the stages that exh14 needs are run (no route search, no
# Model123/181/183/220, only the B4 folds exh14 uses). PART=resnet|b4|hrnet|all selects a lane. As in the original pipeline, the ResNet/ASPP chain
# keeps best.pt by the valid split (valid labels are used for checkpoint selection, never for training).
set -euo pipefail
S=$1; TR=$(cd "$2" && pwd); DATA=$(cd "$3" && pwd); OUT=$4; GPU=${5:-0}
PY=${PYTHON:-python}; export CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$TR/experiments:$TR/experiments/historical_scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT/weights" "$OUT/logs"; W=$(cd "$OUT/weights" && pwd); L=$(cd "$OUT/logs" && pwd)
H="$TR/experiments/historical_scripts"; sd(){ echo $(( $1 + 1000 * S )); }
stage(){ local name=$1 done=$2; shift 2; if [[ -e "$done" ]]; then echo "skip $name"; return; fi; echo "run $name"; "$PY" -u "$@" > "$L/$name.log" 2>&1; [[ -e "$done" ]] || { echo "FAIL $name (no $done)"; exit 1; }; }

resnet_chain(){
stage model6 "$W/model6_unetpp_resnet34/best.pt" "$H/train_models_2_to_6.py" --model-id 6 --data-dir "$DATA" --output-dir "$W/model6_unetpp_resnet34" --epochs 30 --image-size 512 --num-workers 4 --seed $(sd 42)
stage model62 "$W/model62_ft_weighted_safeaug/best.pt" "$H/improve_model6.py" --data-dir "$DATA" --checkpoint "$W/model6_unetpp_resnet34/best.pt" --output-dir "$W/model62_ft_weighted_safeaug" --epochs 20 --batch-size 4 --encoder-lr 5e-6 --decoder-lr 5e-5 --loss-mode weighted --aug-mode safe_plus --select-tta --seed $(sd 42)
stage model97 "$W/model97_native_patch/best.pt" "$H/improve_five_causes.py" --data-dir "$DATA" --checkpoint "$W/model62_ft_weighted_safeaug/best.pt" --output-dir "$W/model97_native_patch" --mode train --patch-size 512 --patches-per-image 6 --val-stride 448 --final-stride 384 --freeze-epochs 3 --finetune-epochs 12 --encoder-lr 5e-6 --decoder-lr 1e-4 --seed $(sd 42)
stage model98 "$W/model98_hard_negative/best.pt" "$H/hard_negative_finetune.py" --data-dir "$DATA" --checkpoint "$W/model97_native_patch/best.pt" --output-dir "$W/model98_hard_negative" --epochs 4 --encoder-lr 1e-6 --decoder-lr 5e-5 --seed $(sd 42)
stage model109 "$W/model109_aspp_native/best.pt" "$H/encoder_structure_search.py" --candidate aspp --data-dir "$DATA" --checkpoint "$W/model98_hard_negative/best.pt" --output-dir "$W/model109_aspp_native" --epochs 4 --patch-size 512 --val-stride 384 --patches-per-image 6 --lr 1.5e-5 --eval-bias -0.2 --seed $(sd 42) --skip-test-inference
stage restart51 "$W/restart51_aspp_768_finetune/best.pt" "$H/encoder_structure_search.py" --candidate aspp --data-dir "$DATA" --checkpoint "$W/model109_aspp_native/best.pt" --output-dir "$W/restart51_aspp_768_finetune" --patch-size 768 --val-stride 576 --tile-batch-size 1 --patches-per-image 4 --epochs 6 --batch-size 2 --lr 8e-6 --eval-bias -0.55 --seed $(sd 123) --skip-test-inference
stage restart59 "$W/restart59_aspp768_seed456/best.pt" "$H/encoder_structure_search.py" --candidate aspp --data-dir "$DATA" --checkpoint "$W/model109_aspp_native/best.pt" --output-dir "$W/restart59_aspp768_seed456" --patch-size 768 --val-stride 576 --tile-batch-size 1 --patches-per-image 4 --epochs 5 --batch-size 2 --lr 8e-6 --eval-bias -0.55 --seed $(sd 456) --skip-test-inference
stage restart61 "$W/restart61_pairwise_margin/best.pt" "$H/pairwise_margin_finetune.py" --data-dir "$DATA" --checkpoint "$W/restart51_aspp_768_finetune/best.pt" --output-dir "$W/restart61_pairwise_margin" --batch-size 1 --num-workers 2 --seed $(sd 42)
stage model194 "$W/model194_multistage_seed2028/best.pt" "$H/model194_multistage_team_schedule.py" --data-dir "$DATA" --checkpoint "$W/model109_aspp_native/best.pt" --output-dir "$W/model194_multistage_seed2028" --stage0-epochs 3 --stage1-epochs 3 --stage2-epochs 0 --stage0-patches 16 --stage1-patches 16 --stage2-patches 8 --stage0-batch 2 --stage1-batch 2 --stage2-batch 1 --encoder-lr 1e-4 --decoder-lr 3e-4 --selection-metric image_mean_miou_exclude --seed $(sd 2028)
stage model195 "$W/model195_native_resume/best.pt" "$H/model194_multistage_team_schedule.py" --data-dir "$DATA" --checkpoint "$W/model194_multistage_seed2028/best.pt" --output-dir "$W/model195_native_resume" --stage0-epochs 0 --stage1-epochs 0 --stage2-epochs 2 --stage2-patches 8 --stage2-batch 2 --eval-patch-size 768 --eval-stride 576 --tile-batch-size 2 --encoder-lr 5e-5 --decoder-lr 1.5e-4 --weight-decay 1e-4 --selection-metric image_mean_miou_exclude --seed $(sd 2028) --num-workers 4
stage model239 "$W/model239_native_stem/best.pt" "$H/model239_native_stem_finetune.py" --data-dir "$DATA" --checkpoint "$W/model195_native_resume/best.pt" --output-dir "$W/model239_native_stem" --epochs 4 --encoder-lr 1e-5 --decoder-lr 3e-5 --native-lr 3e-4 --seed $(sd 239)
stage model242 "$W/model242_native_stem_ema/best_ema.pt" "$H/model242_ema_continuation.py" --data-dir "$DATA" --checkpoint "$W/model239_native_stem/best.pt" --output-dir "$W/model242_native_stem_ema" --epochs 4 --ema-decay 0.99 --encoder-lr 5e-6 --decoder-lr 1.5e-5 --native-lr 1e-4 --seed $(sd 242)
stage model257 "$W/model257_full_image_stats_1x/best.pt" "$H/model253_highres_scale_finetune.py" --data-dir "$DATA" --checkpoint "$W/model242_native_stem_ema/best_ema.pt" --output-dir "$W/model257_full_image_stats_1x" --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 --epochs 4 --scheduler-epochs 4 --encoder-lr 5e-6 --decoder-lr 1.5e-5 --native-lr 1e-4 --weight-decay 1e-4 --val-stride 576 --tta d4 --selection-metric image_mean_miou --seed $(sd 257) --global-representation --num-workers 4
stage model269 "$W/model269_fullstats_1x_extended/best.pt" "$H/model253_highres_scale_finetune.py" --data-dir "$DATA" --checkpoint "$W/model257_full_image_stats_1x/best.pt" --output-dir "$W/model269_fullstats_1x_extended" --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 --epochs 4 --scheduler-epochs 4 --encoder-lr 2.5e-6 --decoder-lr 7.5e-6 --native-lr 5e-5 --weight-decay 1e-4 --val-stride 576 --tta d4 --selection-metric image_mean_miou --seed $(sd 269) --global-representation --num-workers 4

}
CV="$OUT/data_model322_group_cv"
b4(){ local model=$1 code=$2 name=$3 norm=$4 input=$5 scale=$6 tw=$7 sample=$8; shift 8; local f
  for f in "$@"; do mkdir -p "$W/$name/fold$f"
    stage "b4_${model}_fold$f" "$W/$name/fold$f/last.pt" "$code/train.py" --data "$CV" --train_splits "train_fold$f" --out "$W/$name/fold$f" --arch unet --encoder efficientnet-b4 --norm "$norm" --input "$input" --crop 1536 --scale "$scale" --target_width "$tw" --sample "$sample" --bs 2 --iters 16000 --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed $(sd 0) --workers 4 --no_valid_eval
  done; }
r_b4(){
[[ -d "$CV" ]] || "$PY" "$TR/experiments/model322_team_b4/KoMaP/prepare_group_cv.py" --data "$DATA" --manifest "$TR/bundle_assets/training/model303_progress.json" --out "$CV" > "$L/prepare_group_cv.log" 2>&1
b4 322 "$TR/experiments/model322_team_b4/KoMaP" model322_team_b4_group_oof imagenet gray3 3 0 area 3 4
b4 326 "$TR/experiments/model326_hybrid_fullstats/KoMaP" model326_hybrid_fullstats_group_oof image_global zcdf3 3 0 area 1 2
# B4-329 uses the same recipe and the same specimen folds as E's B4 (verified: identical held-out groups, equivalent train.py);
# with SHARE_329=1 (default) it is trained once by retrain_e.sh and linked here by retrain_all.sh.
[[ "${SHARE_329:-1}" == 1 ]] || b4 329 "$TR/experiments/model326_hybrid_fullstats/KoMaP" model329_fullstats_width4000_uniform_group_oof image_global zcdf3 1 4000 uniform 1 3 4
}
r_hrnet(){

stage model336 "$W/model336_hrnet_best/last.pt" "$TR/experiments/model336_hrnet_best/train.py" --data-root "$DATA" --output-dir "$W/model336_hrnet_best" --batch-size 4 --accumulation-steps 2 --epochs 70 --samples-per-image 12 --lr 3e-4 --encoder-lr-mult 1.0 --weight-decay 1e-4 --boundary-head-weight 1.0 --seed $(sd 0) --val-every 5
stage model347 "$W/model347_hrnet_inverse_ce_finetune/last.pt" "$TR/experiments/model336_hrnet_best/train.py" --data-root "$DATA" --output-dir "$W/model347_hrnet_inverse_ce_finetune" --batch-size 4 --accumulation-steps 2 --epochs 12 --samples-per-image 12 --lr 1e-4 --encoder-lr-mult 0.3 --weight-decay 1e-4 --boundary-head-weight 1.0 --seed $(sd 347) --val-every 0 --init-checkpoint "$W/model336_hrnet_best/last.pt" --ce-class-weights 0.22906544 0.49862612 1.39166570 1.88064275
}
case "${PART:-all}" in resnet) resnet_chain;; b4) r_b4;; hrnet) r_hrnet;; all) resnet_chain; r_b4; r_hrnet;; *) echo "PART must be resnet|b4|hrnet|all"; exit 2;; esac
echo "R ${PART:-all} DONE seed=$S -> $W"
