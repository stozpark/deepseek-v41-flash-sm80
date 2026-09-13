#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"

SOURCE_DIR="${SOURCE_DIR:-${ROOT_DIR}/.build/vllm-backport}"
IMAGE="${IMAGE:-$LOCAL_IMAGE}"

command -v git >/dev/null 2>&1 || { echo "ERROR: git not found" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "ERROR: docker not found" >&2; exit 1; }

docker info >/dev/null 2>&1 || {
  echo "ERROR: Docker daemon is not available to this user." >&2
  exit 1
}

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  mkdir -p "$(dirname "$SOURCE_DIR")"
  git clone "$BACKPORT_REPO" "$SOURCE_DIR"
fi

# Reuse a pre-populated checkout in offline environments. Fetch only when the
# pinned commit object is not already present locally.
if ! git -C "$SOURCE_DIR" cat-file -e "${BACKPORT_COMMIT}^{commit}" 2>/dev/null; then
  git -C "$SOURCE_DIR" fetch origin "$BACKPORT_COMMIT"
fi
git -C "$SOURCE_DIR" checkout --detach "$BACKPORT_COMMIT"
ACTUAL_COMMIT="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
[[ "$ACTUAL_COMMIT" == "$BACKPORT_COMMIT" ]] || {
  echo "ERROR: source pin mismatch: $ACTUAL_COMMIT" >&2
  exit 1
}

CORES="$(nproc)"
RAM_GB="$(( $(awk '/MemTotal:/ {print $2}' /proc/meminfo) / 1024 / 1024 ))"
AUTO_JOBS="$(( RAM_GB / 4 ))"
(( AUTO_JOBS > CORES )) && AUTO_JOBS="$CORES"
(( AUTO_JOBS < 2 )) && AUTO_JOBS=2
MAX_JOBS="${MAX_JOBS:-$AUTO_JOBS}"
NVCC_THREADS="${NVCC_THREADS:-1}"

cat <<INFO
[build] source       : $SOURCE_DIR
[build] commit       : $ACTUAL_COMMIT
[build] CUDA         : $CUDA_VERSION
[build] target arch  : sm80 only
[build] image        : $IMAGE
[build] max_jobs     : $MAX_JOBS
INFO

DOCKER_BUILDKIT=1 docker build \
  --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
  --build-arg "BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}" \
  --build-arg "torch_cuda_arch_list=8.0" \
  --build-arg "max_jobs=${MAX_JOBS}" \
  --build-arg "nvcc_threads=${NVCC_THREADS}" \
  --target vllm-openai \
  --progress=plain \
  -f "${SOURCE_DIR}/docker/Dockerfile" \
  -t "$IMAGE" \
  "$SOURCE_DIR"

echo "[ok] built $IMAGE"
