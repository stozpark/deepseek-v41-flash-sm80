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
# Do not publish a manifest if the built image fails package verification.
# Use an independent temporary directory so the mounted port stays read-only.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
python3 "$HERE/image_verify.py" "$TMP/image-manifest.json" --create-from "$SRC"
docker run --rm -v "$HERE:/port:ro" -v "$TMP:/verify:ro" \
  --entrypoint python3 "$TAG" /port/image_verify.py /verify/image-manifest.json
cp "$TMP/image-manifest.json" "$HERE/image-manifest.json"
sha256sum "$HERE/image-manifest.json"
docker image inspect "$TAG" --format '{{.Id}}'
if [[ -n "${SIF_PATH:-}" ]]; then
  apptainer build "$SIF_PATH" "docker-daemon://$TAG"
  sha256sum "$SIF_PATH"
fi
