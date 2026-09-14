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

print("=== direct imports ===")
mods = [
    "vllm.models.deepseek_v4_1",
    "vllm.models.deepseek_v4_1.quant_config",
    "vllm.models.deepseek_v4_1.attention",
    "vllm.models.deepseek_v4_1.common.engram",
    "vllm.models.deepseek_v4_1.nvidia.model",
    "vllm.models.deepseek_v4_1.ampere.ampere_sparse",
    "vllm.v1.attention.backends.mla.indexer",
    "vllm.v1.attention.ops.fp8_sm80",
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

print("=== required sm80 files ===")
import vllm

root = Path(vllm.__file__).resolve().parent
required = [
    root / "v1/attention/ops/fp8_sm80.py",
    root / "models/deepseek_v4_1/ampere/ampere_sparse.py",
    root / "v1/attention/backends/mla/indexer.py",
]
for p in required:
    print(p, p.exists())
    assert p.exists(), p

print("OFFICIAL IMAGE COMPATIBILITY AUDIT OK")
