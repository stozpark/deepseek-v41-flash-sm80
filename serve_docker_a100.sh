#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-lazymio/vllm-backport:v0.13.0-sm80}"
MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash}"
PORT="${PORT:-18005}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

VOL_ARGS=()
MODEL_ARG="$MODEL_PATH"
if [[ -e "$MODEL_PATH" ]]; then
  MODEL_ARG="$(readlink -f "$MODEL_PATH")"
  VOL_ARGS+=(-v "$MODEL_ARG:$MODEL_ARG:ro")
else
  VOL_ARGS+=(-v "${HF_HOME:-$HOME/.cache/huggingface}:/root/.cache/huggingface")
fi

docker run --rm --gpus all --ipc=host \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_ALGO=Ring \
  -e NCCL_PROTO=Simple \
  -p "${PORT}:${PORT}" \
  "${VOL_ARGS[@]}" \
  "$IMAGE" \
  "$MODEL_ARG" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name deepseek-v4.1-flash \
  --tensor-parallel-size 8 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload":true}' \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[6,12,24,48,96,192,384],"max_cudagraph_capture_size":384}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"use_local_argmax_reduction":true}' \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
