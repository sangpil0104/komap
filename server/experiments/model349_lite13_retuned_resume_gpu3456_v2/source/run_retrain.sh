#!/usr/bin/env bash
# lite13 확률 재조정 전체 재학습: 13개 가중치를 학습용 70장(train/)으로 처음부터 다시 만들고, 검증용(채점)·평가용 마스크를 예측한다.
#   bash run_retrain.sh [GPUS]            예: bash run_retrain.sh 0   /   bash run_retrain.sh 0,1,2,3
# 환경 변수
#   DATA   데이터 폴더 (기본 ./data) : train/{images,masks} 70장, valid/{images,masks} 20장, test/images 10장
#   OUT    결과 폴더   (기본 ./runs/retrain) : 다시 실행하면 끝난 단계는 건너뛴다
#   PYTHON 파이썬 실행 파일 (기본 python)
#   SMOKE  1이면 반복 수를 아주 작게 줄여 전체 흐름만 확인한다(성능 없음)
# 모든 경로는 이 폴더 기준 상대 경로이며, 서버 절대 경로를 쓰지 않는다.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
DATA=$(cd "${DATA:-$ROOT/data}" && pwd)
OUT=${OUT:-$ROOT/runs/retrain}; mkdir -p "$OUT/logs"; OUT=$(cd "$OUT" && pwd)
IFS=, read -ra G <<< "${1:-0}"
PY=${PYTHON:-python}; SMOKE=${SMOKE:-0}
T=$ROOT/training; INF=$ROOT/inference; PKG=$INF/parent_package
EXP=$T/train_route/experiments; HS=$EXP/historical_scripts; B=$T/Model349_fusion805/model397_reproduction_bundle
W=$OUT/R; E=$OUT/E; L=$OUT/logs; mkdir -p "$W" "$E"
for d in train/images train/masks valid/images valid/masks; do [[ -d $DATA/$d ]] || { echo "missing $DATA/$d"; exit 1; }; done
[[ $(ls "$DATA/train/images" | wc -l) == 70 ]] || echo "warning: train/images is not 70 images"

# SMOKE=1: epochs -> 1, iterations -> 20 (흐름 확인용)
ep(){ [[ $SMOKE == 1 ]] && echo 1 || echo "$1"; }
it(){ [[ $SMOKE == 1 ]] && echo 20 || echo "$1"; }
# stage NAME DONE_FILE SCRIPT ARGS... : 끝난 단계는 건너뛰고, 실패하면 그 레인을 멈춘다. CWD=폴더 로 작업 폴더 지정
stage(){ local name=$1 done=$2; shift 2
  ( cd "${CWD:-$ROOT}" && "$PY" "$ROOT/../stage.py" "$name" "$done" "$@" ); }


# ---------------------------------------------------------------- R1 restart61, R2 model269 (ResNet34 계열, 이어 학습)
# 원래 파이프라인과 같이 이 계열은 best.pt 선택에 valid 점수를 쓴다(학습에는 쓰지 않음).
lane_resnet(){ export PYTHONPATH="$EXP:$HS${PYTHONPATH:+:$PYTHONPATH}"
stage model6 "$W/model6_unetpp_resnet34/best.pt" "$HS/train_models_2_to_6.py" --model-id 6 --data-dir "$DATA" --output-dir "$W/model6_unetpp_resnet34" --epochs $(ep 30) --image-size 512 --num-workers 4 --seed 42
stage model62 "$W/model62_ft_weighted_safeaug/best.pt" "$HS/improve_model6.py" --data-dir "$DATA" --checkpoint "$W/model6_unetpp_resnet34/best.pt" --output-dir "$W/model62_ft_weighted_safeaug" --epochs $(ep 20) --batch-size 4 --encoder-lr 5e-6 --decoder-lr 5e-5 --loss-mode weighted --aug-mode safe_plus --select-tta --seed 42
stage model97 "$W/model97_native_patch/best.pt" "$HS/improve_five_causes.py" --data-dir "$DATA" --checkpoint "$W/model62_ft_weighted_safeaug/best.pt" --output-dir "$W/model97_native_patch" --mode train --patch-size 512 --patches-per-image 6 --val-stride 448 --final-stride 384 --freeze-epochs $(ep 3) --finetune-epochs $(ep 12) --encoder-lr 5e-6 --decoder-lr 1e-4 --seed 42
stage model98 "$W/model98_hard_negative/best.pt" "$HS/hard_negative_finetune.py" --data-dir "$DATA" --checkpoint "$W/model97_native_patch/best.pt" --output-dir "$W/model98_hard_negative" --epochs $(ep 4) --encoder-lr 1e-6 --decoder-lr 5e-5 --seed 42
stage model109 "$W/model109_aspp_native/best.pt" "$HS/encoder_structure_search.py" --candidate aspp --data-dir "$DATA" --checkpoint "$W/model98_hard_negative/best.pt" --output-dir "$W/model109_aspp_native" --epochs $(ep 4) --patch-size 512 --val-stride 384 --patches-per-image 6 --lr 1.5e-5 --eval-bias -0.2 --seed 42 --skip-test-inference
stage restart51 "$W/restart51_aspp_768_finetune/best.pt" "$HS/encoder_structure_search.py" --candidate aspp --data-dir "$DATA" --checkpoint "$W/model109_aspp_native/best.pt" --output-dir "$W/restart51_aspp_768_finetune" --patch-size 768 --val-stride 576 --tile-batch-size 1 --patches-per-image 4 --epochs $(ep 6) --batch-size 2 --lr 8e-6 --eval-bias -0.55 --seed 123 --skip-test-inference
stage restart61 "$W/restart61_pairwise_margin/best.pt" "$HS/pairwise_margin_finetune.py" --data-dir "$DATA" --checkpoint "$W/restart51_aspp_768_finetune/best.pt" --output-dir "$W/restart61_pairwise_margin" --batch-size 1 --num-workers 2 --seed 42
stage model194 "$W/model194_multistage_seed2028/best.pt" "$HS/model194_multistage_team_schedule.py" --data-dir "$DATA" --checkpoint "$W/model109_aspp_native/best.pt" --output-dir "$W/model194_multistage_seed2028" --stage0-epochs $(ep 3) --stage1-epochs $(ep 3) --stage2-epochs 0 --stage0-patches 16 --stage1-patches 16 --stage2-patches 8 --stage0-batch 2 --stage1-batch 2 --stage2-batch 1 --encoder-lr 1e-4 --decoder-lr 3e-4 --selection-metric image_mean_miou --seed 2028
stage model195 "$W/model195_native_resume/best.pt" "$HS/model194_multistage_team_schedule.py" --data-dir "$DATA" --checkpoint "$W/model194_multistage_seed2028/best.pt" --output-dir "$W/model195_native_resume" --stage0-epochs 0 --stage1-epochs 0 --stage2-epochs $(ep 2) --stage2-patches 8 --stage2-batch 2 --eval-patch-size 768 --eval-stride 576 --tile-batch-size 2 --encoder-lr 5e-5 --decoder-lr 1.5e-4 --weight-decay 1e-4 --selection-metric image_mean_miou --seed 2028 --num-workers 4
stage model239 "$W/model239_native_stem/best.pt" "$HS/model239_native_stem_finetune.py" --data-dir "$DATA" --checkpoint "$W/model195_native_resume/best.pt" --output-dir "$W/model239_native_stem" --epochs $(ep 4) --encoder-lr 1e-5 --decoder-lr 3e-5 --native-lr 3e-4 --seed 239
stage model242 "$W/model242_native_stem_ema/best_ema.pt" "$HS/model242_ema_continuation.py" --data-dir "$DATA" --checkpoint "$W/model239_native_stem/best.pt" --output-dir "$W/model242_native_stem_ema" --epochs $(ep 4) --ema-decay 0.99 --encoder-lr 5e-6 --decoder-lr 1.5e-5 --native-lr 1e-4 --seed 242
stage model257 "$W/model257_full_image_stats_1x/best.pt" "$HS/model253_highres_scale_finetune.py" --data-dir "$DATA" --checkpoint "$W/model242_native_stem_ema/best_ema.pt" --output-dir "$W/model257_full_image_stats_1x" --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 --epochs $(ep 4) --scheduler-epochs $(ep 4) --encoder-lr 5e-6 --decoder-lr 1.5e-5 --native-lr 1e-4 --weight-decay 1e-4 --val-stride 576 --tta d4 --selection-metric image_mean_miou --seed 257 --global-representation --num-workers 4
stage model269 "$W/model269_fullstats_1x_extended/best.pt" "$HS/model253_highres_scale_finetune.py" --data-dir "$DATA" --checkpoint "$W/model257_full_image_stats_1x/best.pt" --output-dir "$W/model269_fullstats_1x_extended" --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 --epochs $(ep 4) --scheduler-epochs $(ep 4) --encoder-lr 2.5e-6 --decoder-lr 7.5e-6 --native-lr 5e-5 --weight-decay 1e-4 --val-stride 576 --tta d4 --selection-metric image_mean_miou --seed 269 --global-representation --num-workers 4
}

# ---------------------------------------------------------------- R7 model347, E3 N3(model414) (HRNet 계열, 기본 HRNet model336에서 출발)
lane_hrnet(){
stage model336 "$W/model336_hrnet_best/last.pt" "$EXP/model336_hrnet_best/train.py" --data-root "$DATA" --output-dir "$W/model336_hrnet_best" --batch-size 4 --accumulation-steps 2 --epochs $(ep 70) --samples-per-image 12 --lr 3e-4 --encoder-lr-mult 1.0 --weight-decay 1e-4 --boundary-head-weight 1.0 --seed 0 --val-every 0
stage model347 "$W/model347_hrnet_inverse_ce_finetune/last.pt" "$EXP/model336_hrnet_best/train.py" --data-root "$DATA" --output-dir "$W/model347_hrnet_inverse_ce_finetune" --batch-size 4 --accumulation-steps 2 --epochs $(ep 12) --samples-per-image 12 --lr 1e-4 --encoder-lr-mult 0.3 --weight-decay 1e-4 --boundary-head-weight 1.0 --seed 347 --val-every 0 --init-checkpoint "$W/model336_hrnet_best/last.pt" --ce-class-weights 0.22906544 0.49862612 1.39166570 1.88064275
# N3 입력: 기존 밝기·Z·CDF를 RGB 8비트로 저장한 이미지(inference/zcdf_input.py와 같은 변환)
stage prepare_n3 "$OUT/data_zcdf_png/done" "$ROOT/tools/make_zcdf_png.py" --data "$DATA" --out "$OUT/data_zcdf_png"
stage model414 "$E/model414_seed414/last.pt" "$EXP/model336_hrnet_best/train.py" --data-root "$OUT/data_zcdf_png" --output-dir "$E/model414_seed414" --device cuda:0 --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --patch-size 512 --input-scale 2 --samples-per-image 12 --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs $(ep 25) --lr 2e-4 --encoder-lr-mult .5 --weight-decay 1e-4 --boundary-head-weight 1 --seed 414 --val-every 0 --init-checkpoint "$W/model336_hrnet_best/last.pt"
}

# ---------------------------------------------------------------- 시편(A번호) 단위 4-fold 분할 (모든 B4·HRNet fold 학습이 공유)
prep_splits(){ stage prepare_splits "$E/splits/model397_split_manifest.json" "$B/scripts/prepare_oof_splits.py" --data-root "$DATA" --output-root "$E/splits"; }


# ---------------------------------------------------------------- R4~R6 / E1~E2: B4 가로 4000 fold 1~4 (+ F 학습용 OOF), BN 고정 추가 학습 fold 1·3
lane_b4(){ local f
for f in 1 2 3 4; do
  mkdir -p "$E/model329/fold$f"
  stage b4_329_fold$f "$E/model329/fold$f/last.pt" "$B/model329/train.py" --data "$E/splits/model329_cv" --train_splits train_fold$f --out "$E/model329/fold$f" \
    --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --target_width 4000 --sample uniform --bs 2 \
    --iters $(it 16000) --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 0 --workers 4 --no_valid_eval
  stage b4_329_fold${f}_oof "$E/model329/fold${f}_oof/probabilities.npz" "$B/model329/predict.py" --data "$E/splits/model329_cv" \
    --split oof_fold$f --runs "$E/model329/fold$f" --out "$E/model329/fold${f}_oof" --save_probs
done
for f in 1 3; do
  mkdir -p "$E/model329_ws/fold$f"
  CWD="$T/Model349_fusion805/b4ws" PYTHONPATH="$PKG/Model349_exh14/oof_design/runtime" OMP_NUM_THREADS=4 \
    stage b4ws_fold$f "$E/model329_ws/fold$f/last.pt" train_bnfreeze.py --data "$E/splits/model329_cv" --train_splits train_fold$f --out "$E/model329_ws/fold$f" \
      --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --scale 1 --target_width 4000 --sample uniform --bs 2 \
      --iters $(it 2000) --lr 3e-5 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 819349 --workers 4 --no_valid_eval --deterministic_workers \
      --init_checkpoint "$E/model329/fold$f/last.pt"
done
}

# ---------------------------------------------------------------- R3: B4 3배 확대 fold 2 (세 밝기 표현, seed 3456, 마지막 가중치 사용)
lane_r326(){ mkdir -p "$W/model326_hybrid_fullstats_group_oof/fold2"
CWD="$EXP/model326_hybrid_fullstats/KoMaP" \
  stage b4_326_fold2 "$W/model326_hybrid_fullstats_group_oof/fold2/last.pt" train_ema.py --data "$E/splits/model329_cv" --train_splits train_fold2 \
    --out "$W/model326_hybrid_fullstats_group_oof/fold2" --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --scale 3 \
    --target_width 0 --sample area --bs 2 --iters $(it 16000) --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 3456 --workers 4 --no_valid_eval \
    --deterministic_workers --ema .999 --ema_start $( [[ $SMOKE == 1 ]] && echo 10 || echo 12000 ) --bn_batches $( [[ $SMOKE == 1 ]] && echo 2 || echo 300 ) --bn_seed 819328
}

# ---------------------------------------------------------------- E4: HRNet 가로 폭 정규화 seed1 fold 1~4, seed2 fold 1~4 (+ F 학습용 OOF; 추론에는 seed1 fold4만 사용)
lane_hr392(){ local s=$1 f run data
for f in 1 2 3 4; do
  run=$E/model392/seed$s/fold$f; data=$E/splits/model392_cv/seed${s}_fold$f; mkdir -p "$run"
  stage hr392_s${s}_f$f "$run/last.pt" "$B/model392/train.py" --data-root "$data" --output-dir "$run" --device cuda:0 \
    --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --encoder-weights imagenet --patch-size 512 --input-scale 2.0 --normalize-width \
    --samples-per-image 12 --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs $(ep 70) --lr 3e-4 --encoder-lr-mult 1.0 \
    --weight-decay 1e-4 --boundary-head-weight 1.0 --seed $s --rng-mode legacy --val-every 0 --resume
  stage hr392_s${s}_f${f}_oof "$E/model392/seed$s/fold${f}_oof/valid_probabilities.npz" "$B/model392/evaluate_image_mean.py" --checkpoint "$run/last.pt" \
    --data-root "$data" --split valid --output-dir "$E/model392/seed$s/fold${f}_oof" --device cuda:0 --batch-size 2 \
    --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4
done
}

# ---------------------------------------------------------------- E5: HRNet 가로 폭 정규화, 학습용 70장 전체 (seed 5, 마지막 epoch)
lane_hr70(){ mkdir -p "$E/model392_train70/seed5"
stage hr392_train70_s5 "$E/model392_train70/seed5/last.pt" "$B/model392/train.py" --data-root "$DATA" --output-dir "$E/model392_train70/seed5" --device cuda:0 \
  --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --encoder-weights imagenet --patch-size 512 --input-scale 2 --normalize-width \
  --samples-per-image 12 --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs $(ep 70) --lr 3e-4 --encoder-lr-mult 1 --weight-decay 1e-4 \
  --boundary-head-weight 1 --rng-mode legacy --seed 5 --val-every 0
}

# ---------------------------------------------------------------- 병렬 실행: 레인을 GPU에 나눠 배정(긴 레인부터)
if [[ ${PLAN_ONLY:-0} == 1 ]]; then
  prep_splits; lane_resnet; lane_hrnet; lane_b4; lane_r326; lane_hr392 1; lane_hr392 2; lane_hr70
else
prep_splits
LANES=(resnet b4 hr1 hr2 hrnet hr70 r326); HOURS=(12 21 9 9 8 3 5)
declare -a LOAD ASSIGN; for gi in "${!G[@]}"; do LOAD[gi]=0; done
for li in "${!LANES[@]}"; do best=0; for gi in "${!G[@]}"; do (( LOAD[gi] < LOAD[best] )) && best=$gi; done; ASSIGN[li]=$best; LOAD[best]=$(( LOAD[best] + HOURS[li] )); done
run_lane(){ local lane=$1 gpu=$2; export CUDA_VISIBLE_DEVICES=$gpu; echo "[$(date +%T)] start $lane on GPU $gpu"
  case $lane in resnet) lane_resnet;; b4) lane_b4;; hr1) lane_hr392 1;; hr2) lane_hr392 2;; hrnet) lane_hrnet;; hr70) lane_hr70;; r326) lane_r326;; esac \
    > "$L/lane_$lane.log" 2>&1; echo "LANE_DONE" >> "$L/lane_$lane.log"; echo "[$(date +%T)] done $lane"; }
pids=()
# GPU마다 하나의 하위 셸에서 배정된 레인을 차례로 실행(한 단계가 실패하면 그 GPU의 나머지 레인은 멈춘다)
for gi in "${!G[@]}"; do ( set -e; for li in "${!LANES[@]}"; do if (( ASSIGN[li] == gi )); then run_lane "${LANES[$li]}" "${G[$gi]}"; fi; done ) & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || true; done
for l in "${LANES[@]}"; do grep -q LANE_DONE "$L/lane_$l.log" 2>/dev/null || { echo "lane $l did not finish"; exit 1; }; done

fi

# ---------------------------------------------------------------- F1: E의 train70 OOF(B4 50% + HRNet seed1 25% + seed2 25%)로 보정기 학습, epoch 8
export CUDA_VISIBLE_DEVICES=${G[0]}
stage F_features "$OUT/F/features/manifest.json" "$T/refiner_retrain/build_features.py" --model397-output "$E" --train "$DATA/train" \
  --palette-config "$PKG/Model349_exh14/configs/model349_lite.yaml" --out "$OUT/F/features"
stage F_train "$OUT/F/training/epoch8.pt" "$T/refiner_retrain/train_refiner.py" --parent-package "$PKG" --features "$OUT/F/features" --out "$OUT/F/training"

# ---------------------------------------------------------------- 추론용 배치로 모으기 (lite13 13개 가중치) 후 예측
[[ ${PLAN_ONLY:-0} == 1 ]] && exit 0
M=$OUT/model; [[ ! -e $M ]] || { echo "Refusing to overwrite $M"; exit 1; }; mkdir -p "$M/weights/R" "$M/weights/E"
ln(){ mkdir -p "$(dirname "$2")"; command ln -sfn "$1" "$2"; }
for k in 1 3 4; do for f in last.pt args.json; do ln "$E/model329/fold$k/$f" "$M/weights/R/model329_fullstats_width4000_uniform_group_oof/fold$k/$f"; done; done
for f in last.pt args.json; do ln "$W/model326_hybrid_fullstats_group_oof/fold2/$f" "$M/weights/R/model326_hybrid_fullstats_group_oof/fold2/$f"; done
ln "$W/restart61_pairwise_margin/best.pt" "$M/weights/R/restart61_pairwise_margin/best.pt"
ln "$W/model269_fullstats_1x_extended/best.pt" "$M/weights/R/model269_fullstats_1x_extended/best.pt"
ln "$W/model347_hrnet_inverse_ce_finetune/last.pt" "$M/weights/R/model347_hrnet_inverse_ce_finetune/last.pt"
for k in 1 3; do for f in last.pt args.json; do ln "$E/model329_ws/fold$k/$f" "$M/weights/E/model329_ws/fold$k/$f"; done; done
for f in last.pt args.json; do ln "$E/model414_seed414/$f" "$M/weights/E/model414_seed414/$f"; done
ln "$E/model392/seed1/fold4/last.pt" "$M/weights/E/model392/seed1/fold4/last.pt"
ln "$E/model392_train70/seed5/last.pt" "$M/weights/E/model392_train70/seed5/last.pt"
ln "$OUT/F/training/epoch8.pt" "$M/weights/E/refiner/epoch8.pt"
for split in valid test; do
  [[ -d $DATA/$split/images ]] || continue
  "$PY" "$ROOT/../timed_predict.py" --source "$ROOT" --weights "$M/weights" --images "$DATA/$split/images" --output "$OUT/pred_$split" --gpus "${1:-0}" --split "$split"
done
stage score_valid "$OUT/pred_valid/metrics.json" "$ROOT/tools/score_masks.py" --labels "$OUT/pred_valid/fusion/predicted_labels" --gt "$DATA/valid/masks" --output "$OUT/pred_valid/metrics.json"
echo "ALL DONE -> $OUT (평가용 마스크: $OUT/pred_test/fusion/predicted_masks)"
