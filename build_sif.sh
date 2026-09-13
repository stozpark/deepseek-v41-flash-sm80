#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"

IMAGE="${IMAGE:-$LOCAL_IMAGE}"
OUT="${1:-${ROOT_DIR}/deepseek-v41-flash-sm80-cu129.sif}"
BUILD_ARGS_STR="${BUILD_ARGS:-}"

command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || {
  echo "ERROR: local image not found: $IMAGE" >&2
  echo "Run ./build_docker_cu129.sh first." >&2
  exit 1
}

if command -v apptainer >/dev/null 2>&1; then
  BUILDER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  BUILDER=singularity
else
  echo "ERROR: apptainer/singularity not found." >&2
  exit 1
fi

BUILD_ARGS=()
if [[ -n "$BUILD_ARGS_STR" ]]; then
  read -r -a BUILD_ARGS <<< "$BUILD_ARGS_STR"
fi

echo "[build] ${BUILDER} build ${OUT} docker-daemon:${IMAGE}"
"$BUILDER" build "${BUILD_ARGS[@]}" "$OUT" "docker-daemon:${IMAGE}"
echo "[ok] $OUT"
