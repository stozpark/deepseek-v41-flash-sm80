#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"

VENDOR_DIR="${ROOT_DIR}/patches/vendor"
OVERLAY_DIR="${ROOT_DIR}/patches/overlay"

required=(
  "${VENDOR_DIR}/BACKPORT_COMMIT"
  "${VENDOR_DIR}/SHA256SUMS"
  "${VENDOR_DIR}/vllm/v1/attention/ops/fp8_sm80.py"
  "${VENDOR_DIR}/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py"
  "${VENDOR_DIR}/vllm/v1/attention/backends/mla/indexer.py"
)
for f in "${required[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: vendored SM80 source missing: $f" >&2; exit 1; }
done

vendor_commit="$(tr -d '[:space:]' < "${VENDOR_DIR}/BACKPORT_COMMIT")"
[[ "$vendor_commit" == "$BACKPORT_COMMIT" ]] || {
  echo "ERROR: vendor pin mismatch: ${vendor_commit} != ${BACKPORT_COMMIT}" >&2
  exit 1
}

(
  cd "${VENDOR_DIR}"
  sha256sum -c SHA256SUMS
)

rm -rf "${OVERLAY_DIR}"
mkdir -p "${OVERLAY_DIR}"
cp -a "${VENDOR_DIR}/." "${OVERLAY_DIR}/"

echo "[ok] offline SM80 overlay prepared from vendored files"
echo "[ok] commit ${vendor_commit}"
echo "[ok] ${OVERLAY_DIR}"
