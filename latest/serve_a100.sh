#!/usr/bin/env bash
# Experimental SM80 bring-up launcher for upstream vLLM v0.32.0.
# This script refuses to claim or assume a completed A100 kernel port.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/VERSION.env"

TP_SIZE="${TP_SIZE:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ID}}"
SIF_PATH="${SIF_PATH:-$(dirname "${ROOT}")/${SIF_NAME}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18005}"
ENABLE_VISION="${ENABLE_VISION:-0}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_EXPERIMENTAL="${ALLOW_EXPERIMENTAL:-0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v41-flash-1017}"

# DeepSeek V4.1-Flash-1017: o_groups=8. Native TP16 is unsupported.
if [[ "${TP_SIZE}" != 8 ]]; then
  echo "ERROR: only TP8 is supported. o_groups=8 makes TP16 invalid." >&2
  echo "For 16 GPUs, run two independent TP8 workers, see serve_two_tp8.sh." >&2
  exit 2
fi
IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "ERROR: exactly eight CUDA_VISIBLE_DEVICES IDs required for TP8: ${GPUS}" >&2
  exit 2
fi
for gpu in "${GPU_ARRAY[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "ERROR: malformed GPU ID: ${gpu}" >&2; exit 2; }
done
if (( $(printf '%s\n' "${GPU_ARRAY[@]}" | sort -u | wc -l) != 8 )); then
  echo "ERROR: CUDA_VISIBLE_DEVICES must contain eight different IDs." >&2
  exit 2
fi
if [[ "${ENABLE_VISION}" != 0 && "${ENABLE_VISION}" != 1 ]]; then
  echo "ERROR: ENABLE_VISION must be 0 or 1." >&2
  exit 2
fi
if (( MAX_NUM_BATCHED_TOKENS < 1 || MAX_NUM_SEQS < 1 || MAX_MODEL_LEN < 1024 )); then
  echo "ERROR: invalid batching or context parameters." >&2
  exit 2
fi

ARGS=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size 8
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --kv-cache-dtype fp8_ds_mla
  --engram-config '{"cpu_offload":true}'
  --attention-config '{"backend":"TRITON_MLA_SPARSE_DSV41"}'
  --tokenizer-mode deepseek_v41
  --reasoning-parser deepseek_v41
  --enable-auto-tool-choice
  --tool-call-parser deepseek_v41
  --disable-custom-all-reduce
  --enforce-eager
)
if [[ "${ENABLE_VISION}" == 0 ]]; then
  ARGS+=(--language-model-only)
else
  ARGS+=(--mm-encoder-tp-mode data)
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  printf '[dry-run] vLLM=%s model=%s cuda-major>=%s GPUs=%s\n' \
      "${VLLM_VERSION}" "${MODEL_PATH}" "${CUDA_MIN_MAJOR}" "${GPUS}"
  printf '[dry-run] '
  printf '%q ' "${ARGS[@]}"
  printf '\n'
  exit 0
fi
if [[ "${ALLOW_EXPERIMENTAL}" != 1 ]]; then
  echo "ERROR: upstream v0.32.0 SM80 inference is NOT yet GPU-validated." >&2
  echo "Set ALLOW_EXPERIMENTAL=1 to explicitly attempt a diagnostic bring-up." >&2
  exit 2
fi
if command -v apptainer >/dev/null 2>&1; then
  RUNNER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNNER=singularity
else
  echo "ERROR: apptainer/singularity is not installed." >&2; exit 2
fi
[[ -f "${SIF_PATH}" ]] || { echo "ERROR: SIF not found: ${SIF_PATH}" >&2; exit 2; }

BIND_ARGS=()
if [[ -e "${MODEL_PATH}" ]]; then
  MODEL_PATH="$(readlink -f "${MODEL_PATH}")"
  ARGS[2]="${MODEL_PATH}"
  BIND_ARGS+=(--bind "${MODEL_PATH}:${MODEL_PATH}")
fi
if [[ -n "${HF_HOME:-}" && -d "${HF_HOME}" ]]; then
  hfhome="$(readlink -f "${HF_HOME}")"
  BIND_ARGS+=(--bind "${hfhome}:${hfhome}")
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"
export PYTHONUNBUFFERED=1
echo "[experimental] Running static + hardware audit (NOT an inference test)"
AUDIT=(python3 /opt/deepseek-sm80/audit_source.py --mode port --runtime)
if [[ -f "${MODEL_PATH}/config.json" ]]; then
  AUDIT+=(--model-config "${MODEL_PATH}/config.json")
fi
"${RUNNER}" exec --nv "${BIND_ARGS[@]}" "${SIF_PATH}" "${AUDIT[@]}"

echo "[experimental] Probe live CUDA SM80 kernels (NOT model validation)"
"${RUNNER}" exec --nv "${BIND_ARGS[@]}" "${SIF_PATH}" \
  python3 /opt/deepseek-sm80/verify_sm80_kernels.py

echo "[experimental] Starting vLLM; no A100 accuracy/soak validation exists yet"
exec "${RUNNER}" exec --nv "${BIND_ARGS[@]}" "${SIF_PATH}" "${ARGS[@]}"
