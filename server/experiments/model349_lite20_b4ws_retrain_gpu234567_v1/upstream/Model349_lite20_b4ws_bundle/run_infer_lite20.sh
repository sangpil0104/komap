#!/usr/bin/env bash
# lite20 (valid20 80.5301%, 20 weights): bash run_infer_lite20.sh WEIGHTS INPUT_IMAGES OUTPUT [GPU] [--allow-retrained]
exec bash "$(cd "$(dirname "$0")" && pwd)/run_infer_variant.sh" lite20 "$@"
