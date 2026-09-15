#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"

OUT="${1:-${ROOT_DIR}/${SIF_NAME}}"
ONLINE_DEF="${ROOT_DIR}/Singularity.def"
VENDOR_DIR="${ROOT_DIR}/patches/vendor"
BUILD_ARGS_STR="${BUILD_ARGS:---fakeroot}"
TMP_DEF=""
TMP_BASE=""

cleanup() {
  [[ -z "$TMP_DEF" || ! -f "$TMP_DEF" ]] || rm -f "$TMP_DEF"
  [[ -z "$TMP_BASE" || ! -f "$TMP_BASE" ]] || rm -f "$TMP_BASE"
}
trap cleanup EXIT

[[ "$OFFICIAL_IMAGE" == "vllm/vllm-openai:deepseekv41-flash-0909" ]] || {
  echo "ERROR: main branch must use the official CUDA 13 image, got: $OFFICIAL_IMAGE" >&2
  exit 1
}
[[ "$CUDA_FAMILY" == 13.0* ]] || {
  echo "ERROR: main branch requires CUDA_FAMILY=13.0.x, got: $CUDA_FAMILY" >&2
  exit 1
}
[[ "$SIF_NAME" == *cu130.sif ]] || {
  echo "ERROR: main branch output name must end in cu130.sif, got: $SIF_NAME" >&2
  exit 1
}
grep -q '^From: vllm/vllm-openai:deepseekv41-flash-0909$' "$ONLINE_DEF" || {
  echo "ERROR: Singularity.def is not pinned to the official CUDA 13 image" >&2
  exit 1
}

for f in \
  "${VENDOR_DIR}/BACKPORT_COMMIT" \
  "${VENDOR_DIR}/SHA256SUMS" \
  "${VENDOR_DIR}/vllm/v1/attention/ops/fp8_sm80.py" \
  "${VENDOR_DIR}/vllm/models/deepseek_v4_1/ampere/ampere_sparse.py"; do
  [[ -f "$f" ]] || {
    echo "ERROR: vendored SM80 sources are incomplete: $f" >&2
    exit 1
  }
done
vendor_commit="$(tr -d '[:space:]' < "${VENDOR_DIR}/BACKPORT_COMMIT")"
[[ "$vendor_commit" == "$BACKPORT_COMMIT" ]] || {
  echo "ERROR: vendor pin mismatch: ${vendor_commit} != ${BACKPORT_COMMIT}" >&2
  exit 1
}
(cd "${VENDOR_DIR}" && sha256sum -c SHA256SUMS >/dev/null)

if command -v apptainer >/dev/null 2>&1; then
  BUILDER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  BUILDER=singularity
else
  echo "ERROR: apptainer/singularity not found." >&2
  exit 1
fi

validate_base_cuda130() {
  local base="$1"
  echo "[check] validating CUDA 13.0 runtime in SIF: $base"
  "$BUILDER" exec "$base" python3 -c '
import sys, torch
cuda = str(torch.version.cuda or "")
print("torch", torch.__version__)
print("torch.version.cuda", cuda)
if not cuda.startswith("13.0"):
    print(f"ERROR: expected CUDA 13.0 runtime, got {cuda!r}", file=sys.stderr)
    raise SystemExit(1)
'
}

read -r -a BUILD_ARGS_ARR <<< "$BUILD_ARGS_STR"
DEF="$ONLINE_DEF"
BASE_SIF_INPUT="${BASE_SIF:-}"

if [[ -n "${BASE_URI:-}" ]]; then
  TMP_BASE="$(mktemp --suffix=.sif)"
  echo "[base] materializing local URI: ${BASE_URI}"
  "${BUILDER}" build "${BUILD_ARGS_ARR[@]}" "$TMP_BASE" "$BASE_URI"
  BASE_SIF_INPUT="$TMP_BASE"
fi

if [[ -n "$BASE_SIF_INPUT" ]]; then
  [[ -f "$BASE_SIF_INPUT" ]] || { echo "ERROR: BASE_SIF not found: $BASE_SIF_INPUT" >&2; exit 1; }
  BASE_SIF_INPUT="$(readlink -f "$BASE_SIF_INPUT")"
  validate_base_cuda130 "$BASE_SIF_INPUT"
  TMP_DEF="$(mktemp --suffix=.def)"
  awk -v base="$BASE_SIF_INPUT" '
    NR == 1 { print "Bootstrap: localimage"; next }
    NR == 2 { print "From: " base; next }
    { print }
  ' "$ONLINE_DEF" > "$TMP_DEF"
  DEF="$TMP_DEF"
  echo "[base] local CUDA 13.0 SIF: ${BASE_SIF_INPUT}"
else
  echo "[base] online registry: ${OFFICIAL_IMAGE}"
fi

echo "[vendor] ${BACKPORT_REPO}@${BACKPORT_COMMIT} (committed in repo)"
echo "[build] ${BUILDER} build ${BUILD_ARGS_STR} ${OUT} ${DEF}"
(
  cd "${ROOT_DIR}"
  "${BUILDER}" build "${BUILD_ARGS_ARR[@]}" "$OUT" "$DEF"
)
validate_base_cuda130 "$OUT"
echo "[ok] ${OUT}"
