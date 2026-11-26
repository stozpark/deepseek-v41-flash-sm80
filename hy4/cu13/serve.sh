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
python3 - <<'PY'
import json, os
from pathlib import Path
import torch
assert torch.cuda.is_available()
assert torch.version.cuda and torch.version.cuda.startswith("13.")
assert torch.cuda.device_count() == 8, "expected eight visible GPUs per node"
assert all(torch.cuda.get_device_capability(i) == (8, 0) for i in range(8))
config = json.loads((Path(os.environ["MODEL_PATH"]) / "config.json").read_text())
t = config.get("text_config", config)
assert "HYV4VLForConditionalGeneration" in config["architectures"]
assert t["num_hidden_layers"] == 78 and t["hidden_size"] == 6144
assert t["n_routed_experts"] == 256 and t["learnable_sink"]
q = config.get("quantization_config") or t.get("quantization_config")
assert q and q.get("quant_method") == "compressed-tensors", "expected official block-FP8 checkpoint"
PY
opts=()
if [[ "$NODE_RANK" == 1 ]]; then opts+=(--headless); fi
exec vllm serve "$MODEL_PATH" \
  --served-model-name hy4 --dtype bfloat16 --kv-cache-dtype bfloat16 \
  --tensor-parallel-size 8 --pipeline-parallel-size 2 \
  --distributed-executor-backend mp --nnodes 2 --node-rank "$NODE_RANK" \
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
