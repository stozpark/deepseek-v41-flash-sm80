#!/usr/bin/env python3
"""CPU-only structural contract regression for inverse-RoPE completion.

This proves the patched return logic takes the fallback branch on SM80;
it does NOT exercise the CUDA kernel or prove numerical parity. Both the
contract check and verify_sm80_numeric.py are required.
"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

REL = "v1/attention/ops/rocm_aiter_mla_sparse.py"


def check(src: Path) -> None:
    tree = ast.parse(src.read_text(encoding="utf8"), filename=str(src))
    defs = [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "rocm_sparse_attn_decode"]
    assert len(defs) == 1, "expected exact sparse decode function"
    body = defs[0].body
    assigns = [
        node for node in body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "fused_inv_rope"
    ]
    assert len(assigns) == 1, "missing one explicit fused-inverse-RoPE predicate"
    final = body[-1]
    assert isinstance(final, ast.Return), "missing inverse-RoPE completion return"
    logic = compile(ast.Expression(assigns[0].value), str(src), "eval")
    result = compile(ast.Expression(final.value), str(src), "eval")
    rows = SimpleNamespace(shape=[3])
    for is_gfx942, is_gfx950, positions, expected in (
        (False, False, None, 0),
        (False, False, object(), 0),  # A100, with requested rotation
        (True, False, None, 0),
        (True, False, object(), 3),
        (False, True, None, 0),
        (False, True, object(), 3),
    ):
        env = {
            "_ON_GFX942": is_gfx942,
            "_ON_GFX950": is_gfx950,
            "inv_rope_positions": positions,
            "output": rows,
        }
        env["fused_inv_rope"] = eval(logic, {"__builtins__": {}}, env)
        actual = eval(result, {"__builtins__": {}}, env)
        assert actual == expected, (
            f"wrong inverse RoPE return on gfx942={is_gfx942}, "
            f"gfx950={is_gfx950}, positions={positions is not None}: "
            f"{actual} != {expected}"
        )
    print("INVERSE_ROPE_FALLBACK_CONTRACT_PASS [6 CPU cases]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    check(args.source_root / REL)
