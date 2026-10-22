#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${1:-${KOMAP_ROOT:-$PWD}}"
declare -a relpaths=(
  "model339_joint_gamma_route_search/results.json"
  "model341_model329_joint_route_search/results.json"
  "model346_model336_into_341_route/results.json"
  "model349_model347_into_346_route/results.json"
  "model355_model349_spatial_audit/results.json"
  "model360_entropy_gated_d4_into_355/results.json"
)

for relpath in "${relpaths[@]}"; do
  src="$BUNDLE_ROOT/results/outputs/$relpath"
  dst="$PROJECT_ROOT/outputs/$relpath"
  [[ -s "$src" ]] || { echo "Missing bundled reference: $src" >&2; exit 2; }
  [[ ! -e "$dst" ]] || { echo "Refusing to overwrite existing file: $dst" >&2; exit 2; }
done

for relpath in "${relpaths[@]}"; do
  dst="$PROJECT_ROOT/outputs/$relpath"
  mkdir -p "$(dirname "$dst")"
  cp "$BUNDLE_ROOT/results/outputs/$relpath" "$dst"
  echo "installed $dst"
done
