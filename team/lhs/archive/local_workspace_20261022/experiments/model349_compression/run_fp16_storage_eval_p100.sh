#!/usr/bin/env bash
set -euo pipefail

# This specific inference process was already using our one P100 when the
# exploratory storage bundle was prepared. Do not overlap another GPU job.
while kill -0 2080395 2>/dev/null; do
  sleep 10
done

memory_used=$(/usr/bin/nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6 | tr -d ' ')
if (( memory_used > 1024 )); then
  echo "GPU 6 is occupied (${memory_used} MiB); refusing to start" >&2
  exit 1
fi

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd /workspace/model349_compress/lite14_fp16_storage

/workspace/venv-komap/bin/python -B -u infer_lite.py \
  --input /workspace/komap_model1/data/valid/images \
  --gt /workspace/komap_model1/data/valid/masks \
  --output /workspace/model349_compress/runs/lite14_fp16_valid20 \
  --gpus 6

/workspace/venv-komap/bin/python -B -u infer_lite.py \
  --input /workspace/komap_model1/data/valid/images \
  --gt /workspace/komap_model1/data/valid/masks \
  --config configs/model349_lite_tent3.yaml \
  --output /workspace/model349_compress/runs/lite14_fp16_tent3_valid20 \
  --gpus 6
