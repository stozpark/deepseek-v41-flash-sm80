#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"

OVERLAY_DIR="${ROOT_DIR}/patches/overlay"
CACHE_DIR="${ROOT_DIR}/.cache/vllm-backport"
SOURCE_DIR="${BACKPORT_SOURCE:-${CACHE_DIR}}"

if [[ -n "${BACKPORT_SOURCE:-}" ]]; then
  [[ -d "${SOURCE_DIR}/.git" ]] || {
    echo "ERROR: BACKPORT_SOURCE is not a git checkout: ${SOURCE_DIR}" >&2
    exit 1
  }
else
  if [[ ! -d "${CACHE_DIR}/.git" ]]; then
    mkdir -p "$(dirname "${CACHE_DIR}")"
    git clone --filter=blob:none --no-checkout "${BACKPORT_REPO}" "${CACHE_DIR}"
  fi
  if ! git -C "${CACHE_DIR}" cat-file -e "${BACKPORT_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${CACHE_DIR}" fetch --depth=1 origin "${BACKPORT_COMMIT}"
  fi
fi

if ! git -C "${SOURCE_DIR}" cat-file -e "${BACKPORT_COMMIT}^{commit}" 2>/dev/null; then
  echo "ERROR: pinned commit ${BACKPORT_COMMIT} is not present in ${SOURCE_DIR}" >&2
  exit 1
fi

PATHS=(
  vllm/models/deepseek_v4
  vllm/models/deepseek_v4_1
  vllm/model_executor/layers/sparse_attn_indexer.py
  vllm/model_executor/warmup/cutedsl_warmup.py
  vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py
  vllm/v1/attention/backends/registry.py
  vllm/v1/attention/backends/mla/indexer.py
  vllm/v1/attention/backends/mla/sparse_swa.py
  vllm/v1/attention/ops/common.py
  vllm/v1/attention/ops/fp8_sm80.py
  vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
)

for path in "${PATHS[@]}"; do
  git -C "${SOURCE_DIR}" cat-file -e "${BACKPORT_COMMIT}:${path}" || {
    echo "ERROR: missing overlay path at pinned commit: ${path}" >&2
    exit 1
  }
done

rm -rf "${OVERLAY_DIR}"
mkdir -p "${OVERLAY_DIR}"
git -C "${SOURCE_DIR}" archive "${BACKPORT_COMMIT}" "${PATHS[@]}" | tar -x -C "${OVERLAY_DIR}"
printf '%s\n' "${BACKPORT_COMMIT}" > "${OVERLAY_DIR}/BACKPORT_COMMIT"

if [[ ! -f "${OVERLAY_DIR}/vllm/v1/attention/ops/fp8_sm80.py" ]] || \
   [[ ! -f "${OVERLAY_DIR}/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py" ]]; then
  echo "ERROR: incomplete SM80 overlay" >&2
  exit 1
fi

echo "[ok] prepared SM80 overlay from ${BACKPORT_COMMIT}"
echo "[ok] ${OVERLAY_DIR}"
