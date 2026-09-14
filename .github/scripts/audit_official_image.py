#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib.metadata as md
from pathlib import Path

import vllm

print("=== versions ===")
for pkg in ("vllm", "torch", "triton", "transformers"):
    try:
        print(pkg, md.version(pkg))
    except Exception as exc:
        print(pkg, "unknown", exc)

root = Path(vllm.__file__).resolve().parent
print("vllm_root", root)

# GPU-less CI validates source/API compatibility only. Physical CUDA import and
# kernel execution are covered by verify_runtime.py on the A100.

print("=== shared SM80 dependency helpers ===")
from vllm.distributed.utils import balanced_row_bounds, balanced_row_counts

assert balanced_row_counts(10, 3) == [4, 3, 3]
assert balanced_row_bounds(5, 15, 0, 3) == (5, 9)
assert balanced_row_bounds(5, 15, 1, 3) == (9, 12)
assert balanced_row_bounds(5, 15, 2, 3) == (12, 15)
print("BALANCED ROW HELPERS OK")

print("=== DeepSeek V4.1 source syntax ===")
source_files = [
    "model_executor/kernels/linear/scaled_mm/cutlass.py",
    "model_executor/kernels/mhc/tilelang.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "model_executor/warmup/cutedsl_warmup.py",
    "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
    "models/deepseek_v4/common/ops/fused_indexer_q.py",
    "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
    "models/deepseek_v4_1/__init__.py",
    "models/deepseek_v4_1/attention.py",
    "models/deepseek_v4_1/sparse_mla.py",
    "models/deepseek_v4_1/compressor.py",
    "models/deepseek_v4_1/common/engram.py",
    "models/deepseek_v4_1/common/ops/cache_utils.py",
    "models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py",
    "models/deepseek_v4_1/common/ops/indexer_k_store.py",
    "models/deepseek_v4_1/amd/rocm.py",
    "models/deepseek_v4_1/ampere/ampere_sparse.py",
    "models/deepseek_v4_1/nvidia/model.py",
    "v1/attention/backends/registry.py",
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/backends/mla/sparse_swa.py",
    "v1/attention/ops/common.py",
    "v1/attention/ops/fp8_sm80.py",
    "v1/attention/ops/mqa_logits_triton.py",
    "v1/attention/ops/rocm_aiter_mla_sparse.py",
    "utils/import_utils.py",
]
for rel in source_files:
    p = root / rel
    assert p.exists(), p
    ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
print("V4.1 SOURCE SYNTAX OK", len(source_files))

print("=== DeepSeek V4.1 registry/routing invariants ===")
registry_text = (root / "model_executor/models/registry.py").read_text(encoding="utf-8")
assert "DeepseekV41ForCausalLM" in registry_text

backend_registry = (root / "v1/attention/backends/registry.py").read_text(encoding="utf-8")
assert "TRITON_MLA_SPARSE_DSV41" in backend_registry

ampere_text = (root / "models/deepseek_v4_1/ampere/ampere_sparse.py").read_text(
    encoding="utf-8"
)
assert 'return "TRITON_MLA_SPARSE_DSV41"' in ampere_text
assert "capability.major == 8" in ampere_text
assert "DeepseekV41AmpereMLAAttention" in ampere_text

v41_model = (root / "models/deepseek_v4_1/nvidia/model.py").read_text(encoding="utf-8")
assert "device_capability.major == 8" in v41_model
assert "AttentionBackendEnum.TRITON_MLA_SPARSE_DSV41" in v41_model
assert "DeepseekV41AmpereMLAAttention" in v41_model
# Production-minimal patch must preserve the official 0909 Engram-DP code.
assert "get_engram_dp_size" in v41_model
assert "gather_engram_hashes" in v41_model
print("V4.1 SM80 ROUTING OK; OFFICIAL ENGRAM-DP MODEL PATH PRESERVED")

print("=== indexer/cache correctness invariants ===")
indexer_text = (root / "v1/attention/backends/mla/indexer.py").read_text(encoding="utf-8")
assert "class DeepseekV41IndexerBackend" in indexer_text
assert "else 128" in indexer_text or "return 128" in indexer_text

mqa_text = (root / "v1/attention/ops/mqa_logits_triton.py").read_text(encoding="utf-8")
assert ".to(tl.int64)" in mqa_text
assert "k_offset < context_len" in mqa_text

v41_cache = (root / "models/deepseek_v4_1/common/ops/cache_utils.py").read_text(
    encoding="utf-8"
)
assert "batch_idx * block_table_stride" in v41_cache
assert "block_table_stride=block_table.stride(0)" in v41_cache
assert "max_blocks_per_seq=block_table.shape[-1]" not in v41_cache
print("V4.1 INDEXER/CACHE GUARDS OK")

print("=== pre-Hopper capability guards ===")
cutlass_text = (root / "model_executor/kernels/linear/scaled_mm/cutlass.py").read_text(
    encoding="utf-8"
)
assert "cutlass_scaled_mm_supports_fp8(compute_capability)" in cutlass_text

mhc_text = (root / "model_executor/kernels/mhc/tilelang.py").read_text(encoding="utf-8")
assert "use_deep_gemm = is_deep_gemm_supported()" in mhc_text
assert "_tilelang_hc_prenorm_gemm(" in mhc_text

import_utils = (root / "utils/import_utils.py").read_text(encoding="utf-8")
assert "def is_cutedsl_supported(" in import_utils
assert "has_device_capability(90)" in import_utils
print("SM80 CAPABILITY GUARDS OK")

print("=== DeepSeek V4.1 Engram pre-SM89 path ===")
engram_text = (root / "models/deepseek_v4_1/common/engram.py").read_text(encoding="utf-8")
assert "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32" in engram_text
assert "values = _decode_fp8_f32(values, False)" in engram_text
assert "weight.view(torch.uint8)" in engram_text
# And, critically, do not regress later official DP/ubatching support while
# adding the A100 byte-decode hunk.
assert "get_engram_dp_group" in engram_text
assert "get_engram_dp_size" in engram_text
assert "dbo_current_ubatch_id" in engram_text
print("V4.1 ENGRAM SM80 DECODE OK; OFFICIAL DP/UBATCHING PRESERVED")

print("=== DeepSeek V4.1 post-0909 correctness hotfixes ===")
swa_text = (root / "v1/attention/backends/mla/sparse_swa.py").read_text(encoding="utf-8")
v41_attn_text = (root / "models/deepseek_v4_1/attention.py").read_text(encoding="utf-8")
assert "def swa_max_image_tokens(" in swa_text
assert "self.max_image_tokens = swa_max_image_tokens(self.vllm_config)" in swa_text
assert "swa_max_image_tokens" in v41_attn_text
assert "self.max_image_tokens = swa_max_image_tokens(vllm_config)" in v41_attn_text
assert "image_width = swa_max_image_tokens(vllm_config)" in v41_cache

tokenizer_text = (root / "tokenizers/deepseek_v41.py").read_text(encoding="utf-8")
assert 'part_type in ("text", "input_text", "output_text")' in tokenizer_text
print("V4.1 POST-0909 HOTFIXES OK")

print("=== V4 shared dependencies actually reached by V4.1 ===")
shared_v4_model = root / "models/deepseek_v4/nvidia/model.py"
assert shared_v4_model.exists()
ast.parse(shared_v4_model.read_text(encoding="utf-8"), filename=str(shared_v4_model))

fused_indexer = root / "models/deepseek_v4/common/ops/fused_indexer_q.py"
fused_indexer_text = fused_indexer.read_text(encoding="utf-8")
assert "fp8_sm80" in fused_indexer_text

inv_rope = root / "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py"
inv_rope_text = inv_rope.read_text(encoding="utf-8")
assert inv_rope_text.count("launch_pdl=launch_pdl") >= 2
print("V4.1 SHARED V4 KERNEL DEPENDENCIES OK")

print("GPU-less audit intentionally skips importing Triton/CUDA model modules")
print("PHYSICAL A100 IMPORT/KERNEL TEST REQUIRED: verify_runtime.py")
print("OFFICIAL DEEPSEEK-V4.1-FLASH MINIMAL-PATCH COMPATIBILITY AUDIT OK")
