#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIN=3470b0db1820f699960d1376223602739a0f4ac3
SRC="$HERE/vllm-hy4-source"
if [[ $# -gt 0 ]]; then SRC="$1"; fi
if [[ -e "$SRC" ]]; then
  echo "Refusing existing source directory: $SRC" >&2; exit 1
fi
git init "$SRC"
git -C "$SRC" remote add origin https://github.com/vllm-project/vllm.git
if ! git -C "$SRC" fetch --depth=1 origin "$PIN"; then
  git -C "$SRC" remote add backport https://github.com/SiluPanda/vllm.git
  git -C "$SRC" fetch --depth=1 backport "$PIN"
fi
git -C "$SRC" checkout --detach "$PIN"
python3 "$HERE/apply_overlay.py" "$SRC" --check-only
python3 "$HERE/apply_overlay.py" "$SRC"
git -C "$SRC" diff --check
git -C "$SRC" status --short
