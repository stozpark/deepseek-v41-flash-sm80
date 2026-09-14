#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.metadata as md
from pathlib import Path

print("=== versions ===")
for pkg in ("vllm", "torch", "triton", "transformers"):
    try:
        print(pkg, md.version(pkg))
    except Exception as exc:
        print(pkg, "unknown", exc)

# GitHub-hosted runners have no NVIDIA driver. vLLM intentionally sets
# vllm.triton_utils.{triton,tl}=None in that environment, which prevents even
# import-only inspection of our Python/Triton modules. Restore the installed
# Triton Python modules for this audit only. We never launch a Triton kernel
# here; physical SM80 execution is covered by verify_runtime.py on the A100.
import triton as _triton
import triton.language as _tl
import vllm.triton_utils as _vtu

if getattr(_vtu, "triton", None) is None:
    _vtu.triton = _triton
if getattr(_vtu, "tl", None) is None:
    _vtu.tl = _tl
print("TRITON IMPORT SHIM OK")

print("=== shared SM80 dependency helpers ===")
from vllm.distributed.utils import balanced_row_bounds, balanced_row_counts

assert balanced_row_counts(10, 3) == [4, 3, 3]
assert balanced_row_bounds(5, 15, 0, 3) == (5, 9)
assert balanced_row_bounds(5, 15, 1, 3) == (9, 12)
assert balanced_row_bounds(5, 15, 2, 3) == (12, 15)
print("BALANCED ROW HELPERS OK")

print("=== direct imports ===")
mods = [
    "vllm.models.deepseek_v4_1",
    "vllm.models.deepseek_v4_1.quant_config",
    "vllm.models.deepseek_v4_1.attention",
    "vllm.models.deepseek_v4_1.common.engram",
    "vllm.models.deepseek_v4_1.nvidia.model",
    "vllm.models.deepseek_v4_1.ampere.ampere_sparse",
    "vllm.model_executor.layers.sparse_attn_indexer",
    "vllm.v1.attention.backends.mla.indexer",
    "vllm.v1.attention.ops.fp8_sm80",
    "vllm.v1.attention.ops.mqa_logits_triton",
    "vllm.v1.attention.ops.rocm_aiter_mla_sparse",
]
for name in mods:
    print("IMPORT", name)
    importlib.import_module(name)
print("DIRECT IMPORTS OK")

print("=== registry ===")
from vllm.model_executor.models.registry import ModelRegistry

registered = ModelRegistry.models.get("DeepseekV41ForCausalLM")
assert registered is not None, "DeepseekV41ForCausalLM is not registered"
cls = registered.load_model_cls()
print("MODEL CLS", cls)
info = registered.inspect_model_cls()
print("MODEL INSPECT OK", info)

print("=== responses-api tokenizer hotfix ===")
from vllm.tokenizers.deepseek_v41 import _normalize_messages

msg = [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]
out = _normalize_messages(msg)
assert out[0]["content"] == "hello", out
msg2 = [{"role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}]
out2 = _normalize_messages(msg2)
assert out2[0]["content"] == "ok", out2
print("RESPONSES API TEXT HOTFIX OK")

print("=== source invariants ===")
import vllm

root = Path(vllm.__file__).resolve().parent
required = [
    root / "v1/attention/ops/fp8_sm80.py",
    root / "v1/attention/ops/mqa_logits_triton.py",
    root / "models/deepseek_v4_1/ampere/ampere_sparse.py",
    root / "v1/attention/backends/mla/indexer.py",
]
for p in required:
    print(p, p.exists())
    assert p.exists(), p

# SM80 paged-indexer long-context correctness (#50576 / #55184 stack).
mqa_text = (root / "v1/attention/ops/mqa_logits_triton.py").read_text()
assert ".to(tl.int64)" in mqa_text
assert "k_offset < context_len" in mqa_text

# --language-model-only must not keep the vision-only SWA width (#56623).
swa_text = (root / "v1/attention/backends/mla/sparse_swa.py").read_text()
v41_attn_text = (root / "models/deepseek_v4_1/attention.py").read_text()
needle = 'language_model_only = bool(getattr(mm_config, "language_model_only", False))'
assert needle in swa_text
assert needle in v41_attn_text

# Pre-Hopper mHC must not unconditionally enter DeepGEMM (#50645).
mhc_text = (root / "model_executor/kernels/mhc/tilelang.py").read_text()
assert "use_deep_gemm = is_deep_gemm_supported()" in mhc_text
assert "_tilelang_hc_prenorm_gemm(" in mhc_text

# SM80 FP8 linear auto-selection must reject unsupported CUTLASS and fall
# through to Marlin (#53376).
cutlass_text = (root / "model_executor/kernels/linear/scaled_mm/cutlass.py").read_text()
assert "cutlass_scaled_mm_supports_fp8(compute_capability)" in cutlass_text

# Narrowed block-table views must use their physical row stride (#55109).
for rel in (
    "models/deepseek_v4/common/ops/cache_utils.py",
    "models/deepseek_v4_1/common/ops/cache_utils.py",
):
    text = (root / rel).read_text()
    assert "batch_idx * block_table_stride" in text, rel
    assert "block_table_stride=block_table.stride(0)" in text, rel
    assert "max_blocks_per_seq=block_table.shape[-1]" not in text, rel

# The pinned backport already contains the launch_pdl correctness fix.
inv_rope = (root / "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py").read_text()
assert inv_rope.count("launch_pdl=launch_pdl") >= 2

print("SM80 SOURCE INVARIANTS OK")
print("OFFICIAL IMAGE COMPATIBILITY AUDIT OK")
