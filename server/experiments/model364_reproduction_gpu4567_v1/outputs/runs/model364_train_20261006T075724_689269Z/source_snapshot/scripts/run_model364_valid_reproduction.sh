#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${KOMAP_ROOT:-$PWD}"
PYTHON="${KOMAP_PYTHON:-python3}"
OUT="${MODEL364_RESULT_PATH:-$PROJECT_ROOT/outputs/model364_reproduction/results.json}"

required=(
  "data/valid/images"
  "data/valid/masks"
  "outputs/model183_metric_calibration/entries.pkl"
  "outputs/model195_native_resume/valid_probs.npz"
  "outputs/model220_native_tent_extended/valid_probs_06.npz"
  "outputs/model269_fullstats_1x_extended/valid_probs.npz"
  "outputs/model322_team_b4_group_oof/valid_ensemble/probabilities.npz"
  "outputs/model326_hybrid_fullstats_group_oof/valid_ensemble/probabilities.npz"
  "outputs/model329_fullstats_width4000_uniform_group_oof/valid_ensemble/probabilities.npz"
  "outputs/model336_hrnet_best/valid_eval/valid_probabilities.npz"
  "outputs/model347_hrnet_inverse_ce_finetune/valid_eval/valid_probabilities.npz"
  "outputs/model351_model336_d4_tta/valid_eval_d4/valid_probabilities.npz"
  "outputs/model361_hrnet_scale3_finetune/valid_eval/valid_probabilities.npz"
  "outputs/model339_joint_gamma_route_search/results.json"
  "outputs/model341_model329_joint_route_search/results.json"
  "outputs/model346_model336_into_341_route/results.json"
  "outputs/model349_model347_into_346_route/results.json"
  "outputs/model355_model349_spatial_audit/results.json"
  "outputs/model360_entropy_gated_d4_into_355/results.json"
)
for relpath in "${required[@]}"; do
  [[ -e "$PROJECT_ROOT/$relpath" ]] || { echo "Missing prerequisite: $PROJECT_ROOT/$relpath" >&2; exit 2; }
done
[[ ! -e "$OUT" ]] || { echo "Refusing to overwrite result: $OUT" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
"$PYTHON" -u "$BUNDLE_ROOT/experiments/model364_model361_entropy_into_360.py" \
  --root "$PROJECT_ROOT" --output "$OUT" \
  2>&1 | tee "${OUT%.json}.log"
