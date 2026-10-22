#!/usr/bin/env bash
# 전체 재학습: 13개 가중치를 학습용 70장(train/)으로 처음부터 다시 만들고, 검증용(채점)·평가용 마스크를 예측한다.
#   bash run_retrain.sh [GPUS]            예: bash run_retrain.sh 0   /   bash run_retrain.sh 0,1,2,3
# 환경 변수
#   DATA   데이터 폴더 (기본 ./data) : train/{images,masks} 70장, valid/{images,masks} 20장, test/images 10장
#   OUT    결과 폴더   (기본 ./runs/retrain) : 다시 실행하면 끝난 단계는 건너뛴다
#   PYTHON 파이썬 실행 파일 (기본 python)
#   SMOKE  1이면 반복 수를 아주 작게 줄여 전체 흐름만 확인한다
# 모든 경로는 이 폴더 기준 상대 경로이며, 서버 절대 경로를 쓰지 않는다.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
DATA=$(cd "${DATA:-$ROOT/data}" && pwd)
OUT=${OUT:-$ROOT/runs/retrain}; mkdir -p "$OUT/logs"; OUT=$(cd "$OUT" && pwd)
IFS=, read -ra G <<< "${1:-0}"
PY=${PYTHON:-python}; SMOKE=${SMOKE:-0}
T=$ROOT/training; INF=$ROOT/inference
EXP=$T/r_models; HS=$EXP/resnet; B=$T/e_models
W=$OUT/R; E=$OUT/E; L=$OUT/logs; mkdir -p "$W" "$E"
for d in train/images train/masks valid/images valid/masks; do [[ -d $DATA/$d ]] || { echo "missing $DATA/$d"; exit 1; }; done
[[ $(ls "$DATA/train/images" | wc -l) == 70 ]] || echo "warning: train/images is not 70 images"

# SMOKE=1: epochs -> 1, iterations -> 20 (흐름 확인용)
ep(){ [[ $SMOKE == 1 ]] && echo 1 || echo "$1"; }
it(){ [[ $SMOKE == 1 ]] && echo 20 || echo "$1"; }
# stage NAME DONE_FILE SCRIPT ARGS... : 끝난 단계는 건너뛰고, 실패하면 그 레인을 멈춘다. CWD=폴더 로 작업 폴더 지정
stage(){ local name=$1 done=$2; shift 2
  if [[ -e "$done" ]]; then echo "skip $name"; return 0; fi
  echo "[$(date +%T)] run $name"
  ( cd "${CWD:-.}" && "$PY" -u "$@" ) > "$L/$name.log" 2>&1 || { echo "FAIL $name (see $L/$name.log)"; exit 1; }
  [[ -e "$done" ]] || { echo "FAIL $name (no $done)"; exit 1; }; }

# ---------------------------------------------------------------- R1·R2 (ResNet34 계열, 이어 학습)
# 원래 파이프라인과 같이 이 계열은 best.pt 선택에 valid 점수를 쓴다(학습에는 쓰지 않음).
lane_R1_R2_resnet(){ export PYTHONPATH="$EXP:$HS${PYTHONPATH:+:$PYTHONPATH}"
stage r1_step1_base "$W/r1_step1_base/best.pt" "$HS/train_resnet_base.py" --model-id 6 --data-dir "$DATA" --output-dir "$W/r1_step1_base" --epochs $(ep 30) --image-size 512 --num-workers 4 --seed 42
stage r1_step2_weighted "$W/r1_step2_weighted/best.pt" "$HS/improve_resnet.py" --data-dir "$DATA" --checkpoint "$W/r1_step1_base/best.pt" --output-dir "$W/r1_step2_weighted" --epochs $(ep 20) --batch-size 4 --encoder-lr 5e-6 --decoder-lr 5e-5 --loss-mode weighted --aug-mode safe_plus --select-tta --seed 42
stage r1_step3_native_patch "$W/r1_step3_native_patch/best.pt" "$HS/improve_five_causes.py" --data-dir "$DATA" --checkpoint "$W/r1_step2_weighted/best.pt" --output-dir "$W/r1_step3_native_patch" --mode train --patch-size 512 --patches-per-image 6 --val-stride 448 --final-stride 384 --freeze-epochs $(ep 3) --finetune-epochs $(ep 12) --encoder-lr 5e-6 --decoder-lr 1e-4 --seed 42
stage r1_step4_hard_negative "$W/r1_step4_hard_negative/best.pt" "$HS/hard_negative_finetune.py" --data-dir "$DATA" --checkpoint "$W/r1_step3_native_patch/best.pt" --output-dir "$W/r1_step4_hard_negative" --epochs $(ep 4) --encoder-lr 1e-6 --decoder-lr 5e-5 --seed 42
stage r1_step5_aspp "$W/r1_step5_aspp/best.pt" "$HS/encoder_structure_search.py" --candidate aspp --data-dir "$DATA" --checkpoint "$W/r1_step4_hard_negative/best.pt" --output-dir "$W/r1_step5_aspp" --epochs $(ep 4) --patch-size 512 --val-stride 384 --patches-per-image 6 --lr 1.5e-5 --eval-bias -0.2 --seed 42 --skip-test-inference
stage r1_step6_aspp768 "$W/r1_step6_aspp768/best.pt" "$HS/encoder_structure_search.py" --candidate aspp --data-dir "$DATA" --checkpoint "$W/r1_step5_aspp/best.pt" --output-dir "$W/r1_step6_aspp768" --patch-size 768 --val-stride 576 --tile-batch-size 1 --patches-per-image 4 --epochs $(ep 6) --batch-size 2 --lr 8e-6 --eval-bias -0.55 --seed 123 --skip-test-inference
stage R1_resnet34_tent "$W/R1_resnet34_tent/best.pt" "$HS/pairwise_margin_finetune.py" --data-dir "$DATA" --checkpoint "$W/r1_step6_aspp768/best.pt" --output-dir "$W/R1_resnet34_tent" --batch-size 1 --num-workers 2 --seed 42
stage r2_step1_multistage "$W/r2_step1_multistage/best.pt" "$HS/multistage_schedule.py" --data-dir "$DATA" --checkpoint "$W/r1_step5_aspp/best.pt" --output-dir "$W/r2_step1_multistage" --stage0-epochs $(ep 3) --stage1-epochs $(ep 3) --stage2-epochs 0 --stage0-patches 16 --stage1-patches 16 --stage2-patches 8 --stage0-batch 2 --stage1-batch 2 --stage2-batch 1 --encoder-lr 1e-4 --decoder-lr 3e-4 --selection-metric image_mean_miou_exclude --seed 2028
stage r2_step2_native_resume "$W/r2_step2_native_resume/best.pt" "$HS/multistage_schedule.py" --data-dir "$DATA" --checkpoint "$W/r2_step1_multistage/best.pt" --output-dir "$W/r2_step2_native_resume" --stage0-epochs 0 --stage1-epochs 0 --stage2-epochs $(ep 2) --stage2-patches 8 --stage2-batch 2 --eval-patch-size 768 --eval-stride 576 --tile-batch-size 2 --encoder-lr 5e-5 --decoder-lr 1.5e-4 --weight-decay 1e-4 --selection-metric image_mean_miou_exclude --seed 2028 --num-workers 4
stage r2_step3_native_stem "$W/r2_step3_native_stem/best.pt" "$HS/native_stem_finetune.py" --data-dir "$DATA" --checkpoint "$W/r2_step2_native_resume/best.pt" --output-dir "$W/r2_step3_native_stem" --epochs $(ep 4) --encoder-lr 1e-5 --decoder-lr 3e-5 --native-lr 3e-4 --seed 239
stage r2_step4_ema "$W/r2_step4_ema/best_ema.pt" "$HS/ema_continuation.py" --data-dir "$DATA" --checkpoint "$W/r2_step3_native_stem/best.pt" --output-dir "$W/r2_step4_ema" --epochs $(ep 4) --ema-decay 0.99 --encoder-lr 5e-6 --decoder-lr 1.5e-5 --native-lr 1e-4 --seed 242
stage r2_step5_fullstats "$W/r2_step5_fullstats/best.pt" "$HS/highres_scale_finetune.py" --data-dir "$DATA" --checkpoint "$W/r2_step4_ema/best_ema.pt" --output-dir "$W/r2_step5_fullstats" --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 --epochs $(ep 4) --scheduler-epochs $(ep 4) --encoder-lr 5e-6 --decoder-lr 1.5e-5 --native-lr 1e-4 --weight-decay 1e-4 --val-stride 576 --tta d4 --selection-metric image_mean_miou --seed 257 --global-representation --num-workers 4
stage R2_resnet34 "$W/R2_resnet34/best.pt" "$HS/highres_scale_finetune.py" --data-dir "$DATA" --checkpoint "$W/r2_step5_fullstats/best.pt" --output-dir "$W/R2_resnet34" --scale 1 --patch-size 1024 --patches-per-image 8 --batch-size 2 --epochs $(ep 4) --scheduler-epochs $(ep 4) --encoder-lr 2.5e-6 --decoder-lr 7.5e-6 --native-lr 5e-5 --weight-decay 1e-4 --val-stride 576 --tta d4 --selection-metric image_mean_miou --seed 269 --global-representation --num-workers 4
}

# ---------------------------------------------------------------- R7, E3 (HRNet 계열, 기본 HRNet에서 출발)
lane_R7_E3_hrnet(){
stage hrnet_base "$W/hrnet_base/last.pt" "$EXP/hrnet/train.py" --data-root "$DATA" --output-dir "$W/hrnet_base" --batch-size 4 --accumulation-steps 2 --epochs $(ep 70) --samples-per-image 12 --lr 3e-4 --encoder-lr-mult 1.0 --weight-decay 1e-4 --boundary-head-weight 1.0 --seed 0 --val-every 0
stage R7_hrnet "$W/R7_hrnet/last.pt" "$EXP/hrnet/train.py" --data-root "$DATA" --output-dir "$W/R7_hrnet" --batch-size 4 --accumulation-steps 2 --epochs $(ep 12) --samples-per-image 12 --lr 1e-4 --encoder-lr-mult 0.3 --weight-decay 1e-4 --boundary-head-weight 1.0 --seed 347 --val-every 0 --init-checkpoint "$W/hrnet_base/last.pt" --ce-class-weights 0.22906544 0.49862612 1.39166570 1.88064275
# E3 입력: 기존 밝기·Z·CDF를 RGB 8비트로 저장한 이미지(inference/e/zcdf_input.py와 같은 변환)
[[ -e $OUT/data_zcdf_png/done ]] || "$PY" "$ROOT/tools/make_zcdf_png.py" --data "$DATA" --out "$OUT/data_zcdf_png"
stage E3_hrnet_zcdf "$E/E3_hrnet_zcdf/last.pt" "$EXP/hrnet/train.py" --data-root "$OUT/data_zcdf_png" --output-dir "$E/E3_hrnet_zcdf" --device cuda:0 --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --patch-size 512 --input-scale 2 --samples-per-image 12 --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs $(ep 25) --lr 2e-4 --encoder-lr-mult .5 --weight-decay 1e-4 --boundary-head-weight 1 --seed 414 --val-every 0 --init-checkpoint "$W/hrnet_base/last.pt"
}

# ---------------------------------------------------------------- 시편(A번호) 단위 4-fold 분할 (모든 B4·HRNet fold 학습이 공유)
prep_splits(){ [[ -d $E/splits/b4_cv ]] || "$PY" "$B/scripts/prepare_oof_splits.py" --data-root "$DATA" --output-root "$E/splits" \
  $( [[ $SMOKE == 1 ]] && echo --subset ) > "$L/prepare_splits.log" 2>&1; }

# ---------------------------------------------------------------- R4~R6 / E1~E2: B4 가로 4000 fold 1~4 (+ F 학습용 OOF), BN 고정 추가 학습 fold 1·3
lane_R4_6_E1_2_b4(){ local f
for f in 1 2 3 4; do
  mkdir -p "$E/b4_w4000/fold$f"
  stage b4_w4000_fold$f "$E/b4_w4000/fold$f/last.pt" "$B/b4/train.py" --data "$E/splits/b4_cv" --train_splits train_fold$f --out "$E/b4_w4000/fold$f" \
    --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --target_width 4000 --sample uniform --bs 2 \
    --iters $(it 16000) --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 0 --workers 4 --no_valid_eval
  stage b4_w4000_fold${f}_oof "$E/b4_w4000/fold${f}_oof/probabilities.npz" "$B/b4/predict.py" --data "$E/splits/b4_cv" \
    --split oof_fold$f --runs "$E/b4_w4000/fold$f" --out "$E/b4_w4000/fold${f}_oof" --save_probs
done
for f in 1 3; do
  mkdir -p "$E/E1_2_b4/fold$f"
  CWD="$T/e_models/b4_bnfreeze" PYTHONPATH="$INF/r/runtime" OMP_NUM_THREADS=4 \
    stage E1_2_b4_fold$f "$E/E1_2_b4/fold$f/last.pt" train_bnfreeze.py --data "$E/splits/b4_cv" --train_splits train_fold$f --out "$E/E1_2_b4/fold$f" \
      --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --scale 1 --target_width 4000 --sample uniform --bs 2 \
      --iters $(it 2000) --lr 3e-5 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 819349 --workers 4 --no_valid_eval --deterministic_workers \
      --init_checkpoint "$E/b4_w4000/fold$f/last.pt"
done
}

# ---------------------------------------------------------------- R3: B4 3배 확대 fold 2 (세 밝기 표현, seed 3456, 마지막 가중치 사용)
lane_R3_b4(){ mkdir -p "$W/R3_b4_x3/fold2"
CWD="$EXP/b4" \
  stage R3_b4_x3_fold2 "$W/R3_b4_x3/fold2/last.pt" train_ema.py --data "$E/splits/b4_cv" --train_splits train_fold2 \
    --out "$W/R3_b4_x3/fold2" --arch unet --encoder efficientnet-b4 --norm image_global --input zcdf3 --crop 1536 --scale 3 \
    --target_width 0 --sample area --bs 2 --iters $(it 16000) --lr 3e-4 --wd 1e-2 --dice_w 1 --cldice_w 0 --seed 3456 --workers 4 --no_valid_eval \
    --deterministic_workers --ema .999 --ema_start $( [[ $SMOKE == 1 ]] && echo 10 || echo 12000 ) --bn_batches $( [[ $SMOKE == 1 ]] && echo 2 || echo 300 ) --bn_seed 819328
}

# ---------------------------------------------------------------- E4: HRNet 가로 폭 정규화 seed1 fold 1~4, seed2 fold 1~4 (+ F 학습용 OOF; 추론에는 seed1 fold4만 사용)
lane_E4_hrnet(){ local s=$1 f run data
for f in 1 2 3 4; do
  run=$E/E4_hrnet/seed$s/fold$f; data=$E/splits/hrnet_cv/seed${s}_fold$f; mkdir -p "$run"
  stage E4_hrnet_seed${s}_fold$f "$run/last.pt" "$B/hrnet/train.py" --data-root "$data" --output-dir "$run" --device cuda:0 \
    --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --encoder-weights imagenet --patch-size 512 --input-scale 2.0 --normalize-width \
    --samples-per-image 12 --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs $(ep 70) --lr 3e-4 --encoder-lr-mult 1.0 \
    --weight-decay 1e-4 --boundary-head-weight 1.0 --seed $s --rng-mode legacy --val-every 0 --resume
  stage E4_hrnet_seed${s}_fold${f}_oof "$E/E4_hrnet/seed$s/fold${f}_oof/valid_probabilities.npz" "$B/hrnet/evaluate_image_mean.py" --checkpoint "$run/last.pt" \
    --data-root "$data" --split valid --output-dir "$E/E4_hrnet/seed$s/fold${f}_oof" --device cuda:0 --batch-size 2 \
    --patch-size 512 --overlap 170 --input-scale 2.0 --tta-mode flip4
done
}

# ---------------------------------------------------------------- E5: HRNet 가로 폭 정규화, 학습용 70장 전체 (seed 5, 마지막 epoch)
lane_E5_hrnet(){ mkdir -p "$E/E5_hrnet_train70/seed5"
stage E5_hrnet_train70 "$E/E5_hrnet_train70/seed5/last.pt" "$B/hrnet/train.py" --data-root "$DATA" --output-dir "$E/E5_hrnet_train70/seed5" --device cuda:0 \
  --architecture UnetPlusPlus --encoder-name tu-hrnet_w48 --encoder-weights imagenet --patch-size 512 --input-scale 2 --normalize-width \
  --samples-per-image 12 --batch-size 4 --accumulation-steps 2 --num-workers 4 --epochs $(ep 70) --lr 3e-4 --encoder-lr-mult 1 --weight-decay 1e-4 \
  --boundary-head-weight 1 --rng-mode legacy --seed 5 --val-every 0
}

# ---------------------------------------------------------------- 병렬 실행: 레인을 GPU에 나눠 배정(긴 레인부터)
prep_splits
LANES=(R1_R2_resnet R4_6_E1_2_b4 E4_hrnet_seed1 E4_hrnet_seed2 R7_E3_hrnet E5_hrnet R3_b4); HOURS=(12 21 9 9 8 3 5)
declare -a LOAD ASSIGN; for gi in "${!G[@]}"; do LOAD[gi]=0; done
for li in "${!LANES[@]}"; do best=0; for gi in "${!G[@]}"; do (( LOAD[gi] < LOAD[best] )) && best=$gi; done; ASSIGN[li]=$best; LOAD[best]=$(( LOAD[best] + HOURS[li] )); done
run_lane(){ local lane=$1 gpu=$2; export CUDA_VISIBLE_DEVICES=$gpu; echo "[$(date +%T)] start $lane on GPU $gpu"
  case $lane in R1_R2_resnet) lane_R1_R2_resnet;; R4_6_E1_2_b4) lane_R4_6_E1_2_b4;; E4_hrnet_seed1) lane_E4_hrnet 1;; E4_hrnet_seed2) lane_E4_hrnet 2;; R7_E3_hrnet) lane_R7_E3_hrnet;; E5_hrnet) lane_E5_hrnet;; R3_b4) lane_R3_b4;; esac \
    > "$L/lane_$lane.log" 2>&1; echo "LANE_DONE" >> "$L/lane_$lane.log"; echo "[$(date +%T)] done $lane"; }
pids=()
# GPU마다 하나의 하위 셸에서 배정된 레인을 차례로 실행(한 단계가 실패하면 그 GPU의 나머지 레인은 멈춘다)
for gi in "${!G[@]}"; do ( set -e; for li in "${!LANES[@]}"; do if (( ASSIGN[li] == gi )); then run_lane "${LANES[$li]}" "${G[$gi]}"; fi; done ) & pids+=($!); done
for p in "${pids[@]}"; do wait "$p" || true; done
for l in "${LANES[@]}"; do grep -q LANE_DONE "$L/lane_$l.log" 2>/dev/null || { echo "lane $l did not finish"; exit 1; }; done

# ---------------------------------------------------------------- F1: E의 train70 OOF(B4 50% + HRNet seed1 25% + seed2 25%)로 보정기 학습, epoch 8
export CUDA_VISIBLE_DEVICES=${G[0]}
[[ -f $OUT/F/features/manifest.json ]] || "$PY" "$T/f_refiner/build_features.py" --e-output "$E" --train "$DATA/train" \
  --palette-config "$INF/config/r_config.yaml" --out "$OUT/F/features" > "$L/F_features.log" 2>&1
[[ -f $OUT/F/training/epoch8.pt ]] || "$PY" "$T/f_refiner/train_refiner.py" --parent-package "$INF" --features "$OUT/F/features" --out "$OUT/F/training" $( [[ $SMOKE == 1 ]] && echo --allow-subset ) > "$L/F_train.log" 2>&1

# ---------------------------------------------------------------- 추론용 배치로 모으기 (가중치 13개) 후 예측
M=$OUT/model; rm -rf "$M"; mkdir -p "$M/weights/R" "$M/weights/E" "$M/weights/F"
ln(){ mkdir -p "$(dirname "$2")"; command ln -sfn "$1" "$2"; }
for k in 1 3 4; do for f in last.pt args.json; do ln "$E/b4_w4000/fold$k/$f" "$M/weights/R/R4_6_b4_w4000/fold$k/$f"; done; done
for f in last.pt args.json; do ln "$W/R3_b4_x3/fold2/$f" "$M/weights/R/R3_b4_x3/fold2/$f"; done
ln "$W/R1_resnet34_tent/best.pt" "$M/weights/R/R1_resnet34_tent/best.pt"
ln "$W/R2_resnet34/best.pt" "$M/weights/R/R2_resnet34/best.pt"
ln "$W/R7_hrnet/last.pt" "$M/weights/R/R7_hrnet/last.pt"
for k in 1 3; do for f in last.pt args.json; do ln "$E/E1_2_b4/fold$k/$f" "$M/weights/E/E1_2_b4/fold$k/$f"; done; done
for f in last.pt args.json; do ln "$E/E3_hrnet_zcdf/$f" "$M/weights/E/E3_hrnet_zcdf/$f"; done
ln "$E/E4_hrnet/seed1/fold4/last.pt" "$M/weights/E/E4_hrnet/seed1/fold4/last.pt"
ln "$E/E5_hrnet_train70/seed5/last.pt" "$M/weights/E/E5_hrnet_train70/seed5/last.pt"
ln "$OUT/F/training/epoch8.pt" "$M/weights/F/F_refiner/epoch8.pt"
for split in valid test; do
  [[ -d $DATA/$split/images ]] || continue
  [[ -e $OUT/pred_$split/completion.json ]] && continue; rm -rf "$OUT/pred_$split"
  ALLOW_RETRAINED=1 bash "$ROOT/run_predict.sh" "$M/weights" "$DATA/$split/images" "$OUT/pred_$split" "${G[0]}"
done
[[ -d $DATA/valid/masks ]] && "$PY" "$ROOT/tools/score_masks.py" --labels "$OUT/pred_valid/fusion/predicted_labels" --gt "$DATA/valid/masks" --output "$OUT/pred_valid/metrics.json"
echo "ALL DONE -> $OUT (평가용 마스크: $OUT/pred_test/fusion/predicted_masks)"
