#!/usr/bin/env bash
# Run inside the built image on BOTH nodes; worker rank 1 is headless.
set -euo pipefail
: "${MODEL_PATH:?set local tencent/Hy4-FP8 checkpoint directory}"
: "${MASTER_ADDR:?set node 0 routable cluster IP}"
: "${NODE_RANK:?set 0 or 1}"
[[ "$NODE_RANK" == 0 || "$NODE_RANK" == 1 ]]
export VLLM_ENABLE_HPC_OPS=0
export VLLM_USE_RUST_FRONTEND=0
export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-37,41}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export MODEL_PATH
# First bring-up: check both nodes independently before allocating full weights.
python3 "$HERE/runtime_preflight.py"
python3 "$HERE/test_kernels.py"
opts=()
if [[ "$NODE_RANK" == 1 ]]; then opts+=(--headless); fi
exec vllm serve "$MODEL_PATH" \
  --served-model-name hy4 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp --data-parallel-backend mp --nnodes 2 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29500}" \
  --attention-config '{"backend":"TRITON_MLA_SPARSE","sparse_mla_force_mqa":true,"sparse_mla_shard_q":false}' \
  --moe-backend marlin --block-size 64 \
  --language-model-only --enforce-eager --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-1024}" --max-num-seqs 1 \
  --enable-auto-tool-choice --reasoning-parser hy_v4 --tool-call-parser hy_v4 \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "${opts[@]}"
