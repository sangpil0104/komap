#!/usr/bin/env bash
# Run lite21 / lite20 / lite19 in parallel (one GPU each) on valid20 (scored) and test10, then compare.
#   bash verify_variants.sh WEIGHTS DATA OUT GPU21 GPU20 GPU19 [LITE23_TEST_LABELS]
# DATA: DATA/valid/{images,masks}, DATA/test/images
# LITE23_TEST_LABELS (optional): an existing lite23 test run's fusion/predicted_labels, to count changed test pixels.
#   Without it, test masks are compared between the variants only.
# Expected valid20 with the published weights: lite21 80.5288%, lite20 80.5304%, lite19 80.5199% (lite23 80.5464%).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); W=$1; DATA=$2; OUT=$3; G21=$4; G20=$5; G19=$6; REF=${7:-}
PY=${PYTHON:-python}
mkdir -p "$OUT"
one(){ local v=$1 g=$2
  bash "$HERE/run_infer_variant.sh" $v "$W" "$DATA/valid/images" "$OUT/$v/valid" $g > "$OUT/$v.valid.log" 2>&1 &&
  bash "$HERE/score.sh" "$OUT/$v/valid/fusion/predicted_labels" "$DATA/valid/masks" "$OUT/$v/valid_metrics.json" >> "$OUT/$v.valid.log" 2>&1 &&
  bash "$HERE/run_infer_variant.sh" $v "$W" "$DATA/test/images" "$OUT/$v/test" $g > "$OUT/$v.test.log" 2>&1 &&
  echo OK > "$OUT/$v.done" || echo FAIL > "$OUT/$v.done"; }
one lite21 $G21 & one lite20 $G20 & one lite19 $G19 & wait
cat "$OUT"/*.done
"$PY" "$HERE/compare_variants.py" "$OUT" "$REF" | tee "$OUT/summary.txt"
