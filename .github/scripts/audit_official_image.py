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

mqa_text = (root / "v1/attention/ops/mqa_logits_triton.py").read_text()
assert ".to(tl.int64)" in mqa_text
assert "k_offset < context_len" in mqa_text

swa_text = (root / "v1/attention/backends/mla/sparse_swa.py").read_text()
v41_attn_text = (root / "models/deepseek_v4_1/attention.py").read_text()
assert 'language_model_only = bool(getattr(mm_config, "language_model_only", False))' in swa_text
assert 'language_model_only = bool(getattr(mm_config, "language_model_only", False))' in v41_attn_text

mhc_text = (root / "model_executor/kernels/mhc/tilelang.py").read_text()
assert "use_deep_gemm = is_deep_gemm_supported()" in mhc_text
assert "_tilelang_hc_prenorm_gemm(" in mhc_text
print("SM80 SOURCE INVARIANTS OK")

print("OFFICIAL IMAGE COMPATIBILITY AUDIT OK")
