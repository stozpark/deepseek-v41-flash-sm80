#!/usr/bin/env python3
"""Static regression: SM80 Engram must pass byte-typed pointers to Triton.

Triton checks argument types before evaluating SOFTWARE_FP8 tl.constexpr.
A float8_e4m3fn GPU or UVA tensor passed directly to @triton.jit triggers:
    ValueError: type fp8e4nv not supported in this architecture
This CPU-only test catches a missing Python caller-side zero-copy uint8 view.
The live verify_sm80_engram.py checks GPU values and the actual wrapper.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

REL = "models/deepseek_v41/common/engram.py"


def node_code(src: str, node: ast.AST) -> str:
    return ast.get_source_segment(src, node) or ""


def run(root: Path) -> None:
    path = root / REL
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    helpers = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_engram_sm80_weight_bytes"
    ]
    if len(helpers) != 1:
        raise AssertionError("expected exactly one SM80 Engram FP8->uint8 helper")
    helper = node_code(src, helpers[0])
    for fragment in (
        "torch.float8_e4m3fn",
        "weight.view(torch.uint8)",
        "byte_view.data_ptr() != weight.data_ptr()",
        "weight.is_contiguous()",
    ):
        if fragment not in helper:
            raise AssertionError(f"missing byte reinterpretation safety check: {fragment}")

    classes = [
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ParallelEngramEmbedding"
    ]
    if len(classes) != 1:
        raise AssertionError("missing ParallelEngramEmbedding")
    lookups = [
        n for n in classes[0].body
        if isinstance(n, ast.FunctionDef) and n.name == "lookup"
    ]
    if len(lookups) != 1:
        raise AssertionError("missing true model lookup method")
    lookup = node_code(src, lookups[0])
    before = lookup.find("weight = _engram_sm80_weight_bytes(weight)")
    call = lookup.find("_engram_lookup_kernel[(grid,)](")
    if not (0 <= before < call):
        raise AssertionError("model lookup must reinterpret FP8 BEFORE Triton argument typing")
    if "SOFTWARE_FP8=software_fp8" not in lookup:
        raise AssertionError("kernel software-FP8 branch must match caller's uint8 view")
    if not all(x in lookup for x in
               ("current_platform.is_cuda()", "is_device_capability_family(80)")):
        raise AssertionError("SM80 byte view must be guarded by CUDA capability")

    kernels = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_engram_lookup_kernel"
    ]
    if len(kernels) != 1:
        raise AssertionError("missing Engram Triton lookup")
    kernel = node_code(src, kernels[0])
    for fragment in ("SOFTWARE_FP8", "_decode_fp8_f32(values, False)",
                     "weight.to(tl.pointer_type(tl.uint8))"):
        if fragment not in kernel:
            raise AssertionError(f"kernel byte decode lost: {fragment}")
    print("ENGRAM_SM80_U8_ARG_ABI_SOURCE_PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    args = p.parse_args()
    run(args.source_root)
