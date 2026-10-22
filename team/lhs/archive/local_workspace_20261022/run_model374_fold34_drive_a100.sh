#!/usr/bin/env bash
set -euo pipefail

# Run from a fresh Colab A100 session after Google Drive has been mounted.
# Stage files expected in /content via `colab upload`:
#   model374_fold34_payload.zip
#   run_model374_fold34_colab_a100.sh
#   model374_drive_checkpoint_sync.py
#   model374_train_oof_error_diagnostic.py

ROOT="${MODEL374_ROOT:-/content/komap_model1}"
DRIVE_ROOT="${MODEL374_DRIVE_ROOT:-/content/drive/MyDrive/KoMaP/model374_fold34_retry_20261001}"
PAYLOAD="${MODEL374_PAYLOAD:-/content/model374_fold34_payload.zip}"
RUNNER="${MODEL374_RUNNER:-/content/run_model374_fold34_colab_a100.sh}"
SYNC="${MODEL374_SYNC_HELPER:-/content/model374_drive_checkpoint_sync.py}"
AUDIT="${MODEL374_OOF_AUDIT:-/content/model374_train_oof_error_diagnostic.py}"

for file in "$PAYLOAD" "$RUNNER" "$SYNC" "$AUDIT"; do
  [[ -s "$file" ]] || { echo "required staged file is missing: $file" >&2; exit 2; }
done
command -v unzip >/dev/null 2>&1 || { echo "unzip is required" >&2; exit 2; }

mkdir -p "$DRIVE_ROOT/source" "$ROOT"
write_probe="$(mktemp "$DRIVE_ROOT/.write_probe.XXXXXX")"
rm -f "$write_probe"

copy_immutable() {
  local source=$1 destination=$2
  if [[ -e "$destination" ]]; then
    cmp -s "$source" "$destination" || {
      echo "refusing to replace a different persistent source file: $destination" >&2
      exit 3
    }
  else
    cp "$source" "$destination"
  fi
}

copy_immutable "$PAYLOAD" "$DRIVE_ROOT/source/model374_fold34_payload.zip"
copy_immutable "$RUNNER" "$DRIVE_ROOT/source/run_model374_fold34_colab_a100.sh"
copy_immutable "$SYNC" "$DRIVE_ROOT/source/model374_drive_checkpoint_sync.py"
copy_immutable "$AUDIT" "$DRIVE_ROOT/source/model374_train_oof_error_diagnostic.py"

unzip -n "$PAYLOAD" -d "$ROOT" >/dev/null
GATE="$ROOT/outputs/model374_hrnet_group_oof/fold1_kill_switch_corrected.json"
[[ -s "$GATE" ]] || { echo "the fixed fold-1 continuation gate is missing from the bundle" >&2; exit 2; }
copy_immutable "$GATE" "$DRIVE_ROOT/source/fold1_kill_switch_corrected.json"
mkdir -p "$ROOT/outputs/model374_hrnet_group_oof" "$ROOT/experiments"
install -m 0644 "$SYNC" "$ROOT/experiments/model374_drive_checkpoint_sync.py"
install -m 0644 "$AUDIT" "$ROOT/experiments/model374_train_oof_error_diagnostic.py"

if [[ "${MODEL374_INSTALL_DEPS:-1}" == "1" ]]; then
  python -m pip install -r "$ROOT/model336_hrnet_best/requirements.txt"
fi
python -c 'import sys, torch; name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA GPU"; print("Compute device:", name); sys.exit(0 if "A100" in name.upper() else "Model374 rerun requires an A100")'

echo "Drive persistence ready: $DRIVE_ROOT"
echo "Fold 3 and 4 will run the fixed Model374 recipe; checkpoints sync every 5 epochs."
MODEL374_DRIVE_ROOT="$DRIVE_ROOT" bash "$RUNNER"
