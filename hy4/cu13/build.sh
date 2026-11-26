#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/vllm-hy4-source"
if [[ $# -gt 0 ]]; then SRC="$1"; fi
PIN=3470b0db1820f699960d1376223602739a0f4ac3
TAG=hy4-sm80:cu13-dev
if [[ -n "${IMAGE_TAG:-}" ]]; then TAG="$IMAGE_TAG"; fi
if [[ ! -d "$SRC/.git" ]]; then bash "$HERE/prepare_source.sh" "$SRC"; fi
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$PIN" ]]
cmp "$HERE/sm80_sparse.py" "$SRC/vllm/models/hy_v4/nvidia/sm80_sparse.py"
cmp "$HERE/sm80_indexer_quant.py" "$SRC/vllm/models/hy_v4/nvidia/sm80_indexer_quant.py"
grep -q 'return HYV4SM80SparseBackend' "$SRC/vllm/models/hy_v4/nvidia/attention.py"
grep -q 'v\[:full_num_tokens\]' "$SRC/vllm/models/hy_v4/nvidia/model.py"
git -C "$SRC" diff --check
# Rebuild the native extensions with the pinned source. Do not mix Python
# modules from a new commit with a possibly ABI-incompatible official image.
# The upstream Dockerfile at this pin defaults to CUDA 13.2.1.
DOCKER_BUILDKIT=1 docker build \
  --target vllm-openai -f "$SRC/docker/Dockerfile" \
  --build-arg "VLLM_BUILD_COMMIT=$PIN" \
  --build-arg "VLLM_IMAGE_TAG=$TAG" \
  -t "$TAG" "$SRC"
docker image inspect "$TAG" --format '{{.Id}}'
if [[ -n "${SIF_PATH:-}" ]]; then
  apptainer build "$SIF_PATH" "docker-daemon://$TAG"
fi
