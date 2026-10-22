#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
GPU_COUNT="${1:-8}"
if [[ ! "$GPU_COUNT" =~ ^(1|2|4|8)$ ]]; then
  echo "Usage: bash scripts/train.sh [1|2|4|8]" >&2
  exit 2
fi
if [[ "$GPU_COUNT" == 1 ]]; then
  python -m alphase.cli.train --config configs/cnn_unet.yaml
else
  python -m torch.distributed.run --standalone --nproc_per_node="$GPU_COUNT" \
    -m alphase.cli.train --config configs/cnn_unet.yaml
fi
python -m alphase.cli.validate --checkpoint runs/cnn_unet/official/best.pt
python -m alphase.cli.predict --checkpoint runs/cnn_unet/official/best.pt
python -m alphase.cli.check_submission
