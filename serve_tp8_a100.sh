#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"
SIF_PATH="${SIF_PATH:-${ROOT_DIR}/${SIF_NAME}}"
MODEL_PATH="${MODEL_PATH:-${MODEL_ID}}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18005}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v4.1-flash}"
TP_SIZE="${TP_SIZE:-8}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"
DISABLE_DSPARK="${DISABLE_DSPARK:-0}"

if command -v apptainer >/dev/null 2>&1; then
  RUNNER=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNNER=singularity
else
  echo "ERROR: apptainer/singularity not found." >&2
  exit 1
fi
[[ -f "$SIF_PATH" ]] || { echo "ERROR: SIF not found: $SIF_PATH" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_PROTO="${NCCL_PROTO:-Simple}"

ARGS=(
  vllm serve "$MODEL_PATH"
  --host "$HOST"
  --port "$PORT"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TP_SIZE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-dtype fp8_ds_mla
  --engram-config '{"cpu_offload":true}'
  --enable-prefix-caching
  --disable-custom-all-reduce
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,24,48,96,192,384],"max_cudagraph_capture_size":384}'
  --tokenizer-mode deepseek_v41
  --enable-auto-tool-choice
  --tool-call-parser deepseek_v41
  --reasoning-parser deepseek_v41
)

if [[ "$DISABLE_DSPARK" != "1" ]]; then
  ARGS+=(--speculative-config '{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true}')
fi
if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  ARGS+=(--enable-expert-parallel)
fi

BIND_ARGS=()
if [[ -e "$MODEL_PATH" ]]; then
  MODEL_PATH="$(readlink -f "$MODEL_PATH")"
  ARGS[2]="$MODEL_PATH"
  BIND_ARGS+=(--bind "$MODEL_PATH:$MODEL_PATH")
fi
if [[ -n "${HF_HOME:-}" && -d "${HF_HOME}" ]]; then
  HF_REAL="$(readlink -f "$HF_HOME")"
  BIND_ARGS+=(--bind "$HF_REAL:$HF_REAL")
fi

echo "[sif]   $SIF_PATH"
echo "[serve] GPUs=$CUDA_VISIBLE_DEVICES TP=$TP_SIZE max_len=$MAX_MODEL_LEN max_seqs=$MAX_NUM_SEQS"
echo "[serve] model=$MODEL_PATH port=$PORT ep=$ENABLE_EXPERT_PARALLEL dspark=$((1-DISABLE_DSPARK))"
exec "$RUNNER" exec --nv "${BIND_ARGS[@]}" "$SIF_PATH" "${ARGS[@]}"
