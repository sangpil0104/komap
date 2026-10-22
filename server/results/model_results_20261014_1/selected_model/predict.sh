#!/usr/bin/env bash
set -euo pipefail
pkg_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source /home/sppark/miniconda3/etc/profile.d/conda.sh
conda activate komap
python - "$pkg_root" <<'PY'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1])
for name,digest in json.loads((r/'files_sha256.json').read_text()).items():
 h=hashlib.sha256()
 with (r/name).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 if h.hexdigest()!=digest:raise ValueError(name)
PY
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 ALLOW_RETRAINED=1
exec bash "$pkg_root/code/run_predict.sh" "$pkg_root/weights" "$1" "$2" "${3:-3,4,5,6}"
