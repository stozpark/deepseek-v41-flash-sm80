#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata as md
import pathlib
import sys


def version(name: str) -> str:
    try:
        return md.version(name)
    except Exception:
        return "unknown"


print("python:", sys.version.replace("\n", " "))
print("torch :", version("torch"))
print("triton:", version("triton"))
print("vllm  :", version("vllm"))

import torch

print("torch.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not (torch.version.cuda or "").startswith("12.9"):
    raise SystemExit(f"ERROR: expected a CUDA 12.9 build, got torch CUDA {torch.version.cuda!r}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(i, torch.cuda.get_device_name(i), cap)
        if cap != (8, 0):
            print(f"WARN: GPU {i} is not SM80", file=sys.stderr)

import vllm

root = pathlib.Path(vllm.__file__).resolve().parent
must_exist = [
    root / "models" / "deepseek_v4_1",
    root / "models" / "deepseek_v4_1" / "ampere" / "ampere_sparse.py",
]
for p in must_exist:
    print("check:", p, "OK" if p.exists() else "MISSING")
    if not p.exists():
        raise SystemExit(2)

print("runtime verification: OK")
