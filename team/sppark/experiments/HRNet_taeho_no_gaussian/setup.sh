#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
alphase_hrnet_python="${HRNET_BASE_PYTHON:-/home/sppark/miniconda3/envs/komap/bin/python}"
if [[ ! -x .venv/bin/python ]]; then
  "$alphase_hrnet_python" -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip freeze > environment.freeze.txt
./run.sh preflight
