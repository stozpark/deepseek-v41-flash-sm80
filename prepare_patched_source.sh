#!/usr/bin/env bash
set -euo pipefail

# Prepare the exact vLLM-backport source revision reported working on 8x A100.
# Do not cherry-pick only a subset onto arbitrary upstream vLLM: V4.1 support
# spans model integration, tokenizer, sparse MLA, FP8 cache, DSpark and warmup.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"
TARGET="${1:-${ROOT_DIR}/vllm-backport-src}"

if [[ -e "$TARGET" ]]; then
  echo "ERROR: target already exists: $TARGET" >&2
  exit 1
fi

git clone "$BACKPORT_REPO" "$TARGET"
git -C "$TARGET" checkout --detach "$BACKPORT_COMMIT"

echo "[ok] prepared: $TARGET"
echo "[pin] $(git -C "$TARGET" rev-parse HEAD)"
