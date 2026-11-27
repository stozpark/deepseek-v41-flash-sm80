#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/VERSION.env"
SRC="${1:-$HERE/vllm-hy4-source}"
TAG="${IMAGE_TAG:-$HY4_IMAGE_TAG}"
if [[ ! -d "$SRC/.git" ]]; then bash "$HERE/prepare_source.sh" "$SRC"; fi
python3 "$HERE/apply_overlay.py" "$SRC" --verify-applied
git -C "$SRC" diff --check
# Build source and C++/CUDA extensions together. Pin the complete upstream
# CUDA 13.2 family; do not mix cu130 wheels with the cu132 builder.
DOCKER_BUILDKIT=1 docker build \
  --target vllm-openai -f "$SRC/docker/Dockerfile" \
  --build-arg "CUDA_VERSION=$HY4_CUDA_TOOLKIT" \
  --build-arg "VLLM_USE_PRECOMPILED=" \
  --build-arg "VLLM_BUILD_COMMIT=$HY4_SOURCE_PIN" \
  --build-arg "VLLM_IMAGE_TAG=$TAG" \
  --label "hy4.sm80.source=$HY4_SOURCE_PIN" \
  -t "$TAG" "$SRC"
# Packaging check needs no GPU. Import the shipped helper rather than the
# host source, and reject stale/unpatched or wrong-CUDA artifacts.
docker run --rm --entrypoint python3 "$TAG" -c '
import importlib.util
import torch
assert torch.version.cuda and torch.version.cuda.startswith("13.2"), torch.version.cuda
for name in ("sm80_sparse", "sm80_indexer_quant", "sink_math"):
    assert importlib.util.find_spec("vllm.models.hy_v4.nvidia." + name), name
print("HY4_IMAGE_PACKAGE=PASS", torch.__version__, torch.version.cuda)
'
docker image inspect "$TAG" --format '{{.Id}}'
if [[ -n "${SIF_PATH:-}" ]]; then
  apptainer build "$SIF_PATH" "docker-daemon://$TAG"
  sha256sum "$SIF_PATH"
fi
