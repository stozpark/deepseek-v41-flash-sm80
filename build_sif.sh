#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${1:-${ROOT_DIR}/deepseek-v41-flash-sm80-cu130.sif}"
DEF="${ROOT_DIR}/Singularity.def"
BUILD_ARGS_STR="${BUILD_ARGS:---fakeroot}"

if command -v apptainer >/dev/null 2>&1; then
  BUILDER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  BUILDER=singularity
else
  echo "ERROR: apptainer/singularity not found." >&2
  exit 1
fi

read -r -a BUILD_ARGS_ARR <<< "$BUILD_ARGS_STR"
echo "[build] ${BUILDER} build ${BUILD_ARGS_STR} ${OUT} ${DEF}"
"${BUILDER}" build "${BUILD_ARGS_ARR[@]}" "$OUT" "$DEF"
echo "[ok] $OUT"
