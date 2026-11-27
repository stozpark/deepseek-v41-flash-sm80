#!/usr/bin/env bash
# Test-only small architecture. Real flagship serving is deliberately separate.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${MODEL_PATH:?set path from make_fixture.py}"
[[ -f "$MODEL_PATH/HY4_DUMMY_FIXTURE" ]]
[[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]
export HY4_FIXTURE_MODE=1 HY4_AUDIT_TRACE=1 VLLM_ENABLE_HPC_OPS=0 VLLM_USE_RUST_FRONTEND=0
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
TP="${TP_SIZE:-1}" PP="${PP_SIZE:-1}" NN="${NNODES:-1}" RANK="${NODE_RANK:-0}"
if [[ "$PP" == 1 ]]; then export VLLM_PP_LAYER_PARTITION=4
else export VLLM_PP_LAYER_PARTITION="${VLLM_PP_LAYER_PARTITION:-2,2}"; fi
opts=()
if [[ "$NN" == 2 ]]; then
  : "${MASTER_ADDR:?set cluster IP}"
  opts+=(--nnodes 2 --node-rank "$RANK" --master-addr "$MASTER_ADDR" --master-port "${MASTER_PORT:-29500}")
  if [[ "$RANK" == 1 ]]; then opts+=(--headless); fi
fi
# Always verify the exact package on the physical GPU host.
python3 "$HERE/image_verify.py" "${HY4_IMAGE_MANIFEST:-$HERE/image-manifest.json}" --gpu
exec vllm serve "$MODEL_PATH" --load-format dummy --seed 1234 \
  --served-model-name hy4 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" \
  --distributed-executor-backend mp --data-parallel-backend mp \
  --attention-config '{"backend":"TRITON_MLA_SPARSE","sparse_mla_force_mqa":true,"sparse_mla_shard_q":false}' \
  --moe-backend marlin --linear-backend marlin --block-size 64 \
  --worker-cls audited_worker.AuditedWorker --language-model-only --enforce-eager \
  --no-enable-prefix-caching --disable-custom-all-reduce \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.30}" \
  --max-model-len "${MAX_MODEL_LEN:-4096}" --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-1024}" \
  --max-num-seqs "${MAX_NUM_SEQS:-2}" --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "${opts[@]}"
