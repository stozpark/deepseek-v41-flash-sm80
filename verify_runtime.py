#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import torch
import vllm

root = Path(vllm.__file__).resolve().parent
required = [
    root / "v1/attention/ops/fp8_sm80.py",
    root / "v1/attention/ops/mqa_logits_triton.py",
    root / "models/deepseek_v4/ampere/ampere_sparse.py",
    root / "models/deepseek_v4_1/ampere/ampere_sparse.py",
    root / "v1/attention/backends/mla/indexer.py",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"ERROR: missing SM80 overlay files: {missing}")

from vllm.v1.attention.backends.registry import AttentionBackendEnum

if not hasattr(AttentionBackendEnum, "TRITON_MLA_SPARSE_DSV41"):
    raise SystemExit("ERROR: TRITON_MLA_SPARSE_DSV41 is not registered")

print("vllm", md.version("vllm"))
print("torch", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
try:
    print("triton", md.version("triton"))
except Exception:
    pass
print("vllm_root", root)

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable; run this inside the SIF with --nv")

for idx in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(idx)
    cap = torch.cuda.get_device_capability(idx)
    print(f"gpu[{idx}]={name} capability={cap[0]}.{cap[1]}")
    if cap[0] != 8:
        raise SystemExit(f"ERROR: GPU {idx} is not SM8x: {cap}")

device = torch.device("cuda:0")

from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
    DeepseekV41AmpereMLASparseBackend,
)

print("backend", DeepseekV41AmpereMLASparseBackend.get_name())

# A100 must not auto-select CUTLASS FP8. It should fall through to the Marlin
# fallback used on GPUs without native FP8 tensor-core support (#53376).
from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
    CutlassFP8ScaledMMLinearKernel,
)

cutlass_ok, cutlass_reason = CutlassFP8ScaledMMLinearKernel.is_supported(80)
print("cutlass_fp8_sm80", cutlass_ok, cutlass_reason)
if cutlass_ok:
    raise SystemExit("ERROR: CUTLASS FP8 incorrectly reports SM80 support")

# Compile and execute the actual SM80 paged-indexer Triton fallback. A one-token
# context with a width-65 logits row is intentionally chosen to catch both the
# long-context physical-address path and the tail-store mask regression.
from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton

B, NEXT_N, H, D = 1, 1, 16, 128
BLOCK_SIZE = 64
q = torch.zeros((B, NEXT_N, H, D), dtype=torch.float8_e4m3fn, device=device)
kv_cache = torch.zeros(
    (1, BLOCK_SIZE, 1, D + 4), dtype=torch.uint8, device=device
)
weights = torch.ones((B * NEXT_N, H), dtype=torch.float32, device=device)
context_lens = torch.tensor([1], dtype=torch.int32, device=device)
block_tables = torch.zeros((B, 2), dtype=torch.int32, device=device)
logits = fp8_paged_mqa_logits_triton(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    max_model_len=65,
)
torch.cuda.synchronize()
if not torch.isfinite(logits[0, 0]):
    raise SystemExit("ERROR: SM80 paged-MQA live logit is not finite")
if not torch.isneginf(logits[0, 1:]).all():
    raise SystemExit("ERROR: SM80 paged-MQA tail mask corrupted inactive logits")
print("paged_mqa_sm80_tail_mask: OK")

# Reproduce the narrowed block-table case from vLLM #55109. The active view
# has width 1 but retains a backing stride of 4. Before the fix request 2+
# addressed the wrong row and could issue an illegal memory access. Compare the
# narrowed view against its contiguous copy; they must be bit-identical.
from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)

NUM_REQS = 4
HEAD_DIM = 512
NOPE_DIM = 448
SCALE_DIM = 8
HEAD_BYTES = NOPE_DIM + (HEAD_DIM - NOPE_DIM) * 2 + SCALE_DIM
GATHER_BLOCK_SIZE = 256
compressed_kv = torch.randn(
    NUM_REQS, HEAD_DIM, dtype=torch.bfloat16, device=device
)
physical_blocks = torch.tensor([3, 1, 0, 2], dtype=torch.int32, device=device)
block_table_storage = torch.full(
    (NUM_REQS, 4), -1000000, dtype=torch.int32, device=device
)
block_table_view = block_table_storage[:, :1]
block_table_view[:, 0] = physical_blocks
if block_table_view.stride(0) == block_table_view.shape[-1]:
    raise SystemExit("ERROR: strided block-table test did not create a narrowed view")
slot_mapping = physical_blocks.to(torch.int64) * GATHER_BLOCK_SIZE
k_cache = torch.empty(
    NUM_REQS,
    GATHER_BLOCK_SIZE,
    HEAD_BYTES,
    dtype=torch.uint8,
    device=device,
)
quantize_and_insert_k_cache(
    compressed_kv,
    k_cache.view(NUM_REQS, -1),
    slot_mapping,
    GATHER_BLOCK_SIZE,
)
seq_lens = torch.ones(NUM_REQS, dtype=torch.int32, device=device)
actual = torch.empty(
    NUM_REQS, 1, HEAD_DIM, dtype=torch.bfloat16, device=device
)
expected = torch.empty_like(actual)
dequantize_and_gather_k_cache(
    actual,
    k_cache,
    seq_lens,
    None,
    block_table_view,
    GATHER_BLOCK_SIZE,
    0,
)
dequantize_and_gather_k_cache(
    expected,
    k_cache,
    seq_lens,
    None,
    block_table_view.contiguous(),
    GATHER_BLOCK_SIZE,
    0,
)
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print("strided_block_table_gather: OK")

# Static guard for the post-0909 text-only SWA fix.
swa_text = (root / "v1/attention/backends/mla/sparse_swa.py").read_text()
if 'language_model_only = bool(getattr(mm_config, "language_model_only", False))' not in swa_text:
    raise SystemExit("ERROR: --language-model-only SWA-width hotfix missing")

print("SM80 runtime verification: OK")
