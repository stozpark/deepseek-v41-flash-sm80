#!/usr/bin/env python3
"""Source regression for SM80 software-FP8 Triton pointer signatures.

SM80's Triton JIT rejects ptr<fp8e4nv> before a tl.constexpr software
conversion can run. All optional FP8 writers must reinterpret the POINTER
ARGUMENT to uint8 on Python-side and preserve the original tensor return type.
This does not prove full GPU arithmetic correctness.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def check(root: Path) -> None:
    checks={
        "models/deepseek_v41/common/ops/fused_compress_quant_cache.py": (
            "kv_cache.view(torch.uint8)",
            "store_fp8 and current_platform.is_cuda()",
            "_rope_plain_insert_kernel[(num_tokens,)](",
        ),
        "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py": (
            "out_buf.view(torch.uint8)",
            "if quantize and current_platform.is_cuda()",
            "if compile_key.quantize and current_platform.is_cuda()",
            "torch.uint8",
        ),
        "models/deepseek_v41/common/engram.py": (
            "weight.view(torch.uint8)",
            "weight = _engram_sm80_weight_bytes(weight)",
            "_engram_lookup_kernel[(grid,)](",
        )
    }
    for filename,fragments in checks.items():
        f=root/filename
        source=f.read_text()
        ast.parse(source,filename=filename)
        missing=[frag for frag in fragments if frag not in source]
        if missing:
            raise AssertionError(f"{filename}: missing SM80 byte ABI gate(s): {missing}")
    print("SM80_E4M3FN_PYTHON_JIT_POINTER_ABI_SOURCE_PASS (3 writer paths)")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    args=p.parse_args()
    check(args.source_root)
