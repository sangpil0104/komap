#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$EXPERIMENT_DIR"

GPU_COUNT="${1:-8}"
if [[ ! "$GPU_COUNT" =~ ^(1|2|4|8)$ ]]; then
  echo "Usage: bash run_suite.sh [1|2|4|8]" >&2
  exit 2
fi

# Detect komap python environment
if [[ -x "/home/sppark/miniconda3/envs/komap/bin/python" ]]; then
  PYTHON_EXE="/home/sppark/miniconda3/envs/komap/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_EXE="$(command -v python)"
else
  echo "Error: Python interpreter not found." >&2
  exit 1
fi

echo "=========================================================="
echo "Starting ResNet34 U-Net++ + Residual ASPP Suite"
echo "Experiment Dir : $EXPERIMENT_DIR"
echo "Using GPUs     : $GPU_COUNT"
echo "Python         : $PYTHON_EXE"
echo "=========================================================="

# 1. Training
echo ""
echo ">> [1/4] Running Multi-Stage Training (Effective Batch: 8)..."
"$PYTHON_EXE" "$EXPERIMENT_DIR/run.py" train --gpus "$GPU_COUNT"

# 2. Plain & D4 Evaluation
echo ""
echo ">> [2/4] Running Validation Evaluation (Plain & D4 TTA)..."
"$PYTHON_EXE" "$EXPERIMENT_DIR/run.py" evaluate
"$PYTHON_EXE" "$EXPERIMENT_DIR/run.py" evaluate --d4

# 3. Test Predictions
echo ""
echo ">> [3/4] Generating Test Mask PNGs (10 images, RGBA exact palette)..."
"$PYTHON_EXE" "$EXPERIMENT_DIR/run.py" predict --d4

# 4. Generate Final Report
echo ""
echo ">> [4/4] Updating Results Report (results/model_results_0921_1.md)..."
"$PYTHON_EXE" "$EXPERIMENT_DIR/run.py" report

echo ""
echo "=========================================================="
echo "Experiment Completed Successfully!"
echo "Check results at: /Data3/sppark/komap/results/model_results_0921_1.md"
echo "Predictions at  : $EXPERIMENT_DIR/outputs/predictions/"
echo "=========================================================="

