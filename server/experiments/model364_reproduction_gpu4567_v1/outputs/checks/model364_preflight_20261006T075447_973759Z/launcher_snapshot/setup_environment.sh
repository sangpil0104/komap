#!/usr/bin/env bash
set -euo pipefail
model364_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/sppark/miniconda3/etc/profile.d/conda.sh
conda activate komap
export PYTHONDONTWRITEBYTECODE=1
"$CONDA_PREFIX/bin/python" -B - <<'PY'
import importlib.metadata as m
import sys
from pathlib import Path
assert Path(sys.prefix).name == 'komap', sys.prefix
for package in ['torch', 'torchvision', 'numpy', 'Pillow', 'scipy', 'tqdm',
                'segmentation-models-pytorch', 'timm', 'albumentations', 'opencv-python-headless']:
    print(f'{package}=={m.version(package)}')
import torch, torchvision, segmentation_models_pytorch, timm, albumentations, cv2
assert torch.cuda.is_available(), 'CUDA unavailable'
print('Existing conda komap dependencies and CUDA OK')
PY
"$CONDA_PREFIX/bin/python" -m pip freeze > "$model364_root/reports/environment.freeze.txt"
exec "$model364_root/run.sh" check
