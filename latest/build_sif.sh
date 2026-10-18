#!/usr/bin/env bash
# Experimental vLLM 0.32.0 A100 build; produces stage-1 source integration only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/latest/VERSION.env"

if command -v apptainer >/dev/null 2>&1; then
  BUILDER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  BUILDER=singularity
else
  echo "ERROR: apptainer/singularity is required." >&2
  exit 2
fi
for f in latest/Singularity.def latest/port_sm80_stage1.py latest/port_sm80_numeric.py latest/port_sm80_indexer.py latest/test_inverse_rope_contract.py \
         latest/ampere_sparse.py latest/audit_source.py latest/verify_sm80_kernels.py latest/verify_sm80_numeric.py latest/verify_sm80_indexer.py \
         patches/vendor/vllm/v1/attention/ops/fp8_sm80.py \
         patches/vendor/vllm/v1/attention/ops/mqa_logits_triton.py; do
  [[ -s "${ROOT}/${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 2; }
done
grep -qx 'From: vllm/vllm-openai:v0.32.0' "${ROOT}/latest/Singularity.def" || {
  echo "ERROR: unpinned upstream container"; exit 2;
}
OUT="${1:-${ROOT}/${SIF_NAME}}"
BASE_SIF="${BASE_SIF:-}"
DEF="${ROOT}/latest/Singularity.def"
TMP_DEF=""
cleanup() { [[ -z "${TMP_DEF}" ]] || rm -f "${TMP_DEF}"; }
trap cleanup EXIT

if [[ -n "${BASE_SIF}" ]]; then
  [[ -s "${BASE_SIF}" ]] || { echo "ERROR: BASE_SIF missing"; exit 2; }
  BASE_SIF="$(readlink -f "${BASE_SIF}")"
  "${BUILDER}" exec "${BASE_SIF}" python3 -c '
import torch, importlib.metadata as md
assert md.version("vllm") == "0.32.0"
assert int(str(torch.version.cuda or "0").split(".")[0]) >= 13
'
  TMP_DEF="$(mktemp --suffix=.def)"
  awk -v base="${BASE_SIF}" '
    NR == 1 { print "Bootstrap: localimage"; next }
    NR == 2 { print "From: " base; next }
    { print }
  ' "${DEF}" > "${TMP_DEF}"
  DEF="${TMP_DEF}"
fi

BUILD_ARGS="${BUILD_ARGS:---fakeroot}"
read -r -a EXTRA_ARGS <<< "${BUILD_ARGS}"
echo "[EXPERIMENTAL] Source-port stage1 SIF, NOT a GPU-tested serving image"
echo "vLLM=${VLLM_VERSION} image=${VLLM_IMAGE} cuda>=${CUDA_MIN_MAJOR}"
(
  cd "${ROOT}"
  "${BUILDER}" build "${EXTRA_ARGS[@]}" "${OUT}" "${DEF}"
)
"${BUILDER}" exec "${OUT}" python3 /opt/deepseek-sm80/audit_source.py --mode port
echo "STAGE1_BUILD_PASS: ${OUT}"
echo "WARNING: SM80 FP8 cache/indexer/Engram kernel integration is incomplete."
