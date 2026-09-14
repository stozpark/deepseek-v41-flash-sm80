#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as md
from pathlib import Path

import torch
import vllm

root = Path(vllm.__file__).resolve().parent
required = [
    root / "v1/attention/ops/fp8_sm80.py",
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

from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
    DeepseekV41AmpereMLASparseBackend,
)

print("backend", DeepseekV41AmpereMLASparseBackend.get_name())
print("SM80 runtime verification: OK")
