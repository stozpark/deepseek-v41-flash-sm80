#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"
OUT="${1:-${ROOT_DIR}/${SIF_NAME}}"
DEF="${ROOT_DIR}/Singularity.def"
BUILD_ARGS_STR="${BUILD_ARGS:---fakeroot}"

if [[ "${SKIP_OVERLAY_PREPARE:-0}" != "1" ]]; then
  "${ROOT_DIR}/prepare_sm80_overlay.sh"
else
  [[ -f "${ROOT_DIR}/patches/overlay/vllm/v1/attention/ops/fp8_sm80.py" ]] || {
    echo "ERROR: SKIP_OVERLAY_PREPARE=1 but patches/overlay is missing." >&2
    exit 1
  }
fi

if command -v apptainer >/dev/null 2>&1; then
  BUILDER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  BUILDER=singularity
else
  echo "ERROR: apptainer/singularity not found." >&2
  exit 1
fi

read -r -a BUILD_ARGS_ARR <<< "$BUILD_ARGS_STR"
echo "[base]  ${OFFICIAL_IMAGE}"
echo "[patch] ${BACKPORT_REPO}@${BACKPORT_COMMIT}"
echo "[build] ${BUILDER} build ${BUILD_ARGS_STR} ${OUT} ${DEF}"
(
  cd "${ROOT_DIR}"
  "${BUILDER}" build "${BUILD_ARGS_ARR[@]}" "$OUT" "$DEF"
)
echo "[ok] ${OUT}"
