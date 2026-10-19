#!/usr/bin/env python3
"""Source-only contract for Ampere WO_A output projection.

The chosen CUDA SM80 attention class derives from the ROCm BF16/einsum
correctness implementation (DeepseekV41ROCMAiterMLAAttention._o_proj).
It must NOT set is_bmm=True while using that forward. BMM-specific
weight repacking is valid only after providing/testing a BMM forward.
This structural test does not validate real A100 output numerics.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path

UPSTREAM="models/deepseek_v41/amd/rocm.py"
AMPERE="models/deepseek_v41/ampere/ampere_sparse.py"

def run(root: Path) -> None:
    sm80=(root/AMPERE).read_text()
    rocm=(root/UPSTREAM).read_text()
    sm80_ast=ast.parse(sm80,filename=AMPERE)
    sm80_classes=[
        n for n in sm80_ast.body
        if isinstance(n,ast.ClassDef) and n.name=="DeepseekV41AmpereMLAAttention"
    ]
    if len(sm80_classes)!=1:
        raise AssertionError("Missing SM80 attention class")
    cls=sm80_classes[0]
    if not any(isinstance(base,ast.Name)
               and base.id=="DeepseekV41ROCMAiterMLAAttention"
               for base in cls.bases):
        raise AssertionError("SM80 no longer inherits ROCm correctness reference")
    overridden={
        node.name for node in cls.body
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
    }
    if "_o_proj" in overridden:
        raise AssertionError("SM80 changed WO_A projector; requires separate GPU parity")
    if "self.wo_a.is_bmm = True" in sm80:
        raise AssertionError("BMM post-load flag not compatible with inherited einsum forward")
    if "self.wo_a.is_bmm = False" not in rocm:
        raise AssertionError("Upstream ROCm WO_A loading contract changed")
    if "rocm_inv_rope_einsum(" not in rocm:
        raise AssertionError("Upstream ROCm BF16/einsum projection no longer present")
    print("SM80_WOA_BF16_EINSUM_WEIGHT_LAYOUT_CONTRACT_PASS")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source-root",type=Path,required=True)
    args=p.parse_args()
    run(args.source_root)
