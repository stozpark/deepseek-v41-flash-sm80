#!/usr/bin/env python3
"""No-GPU regression for old SM80 MQA tuning options on pinned vLLM 0.32.0.

The vendored fallback originates in a version that extended vllm.envs.
The three extensions are absent from upstream v0.32.0; the fallback must
read optional os.environ settings directly while preserving numerical defaults.
This test evaluates only the *pure* helper functions, with no CUDA/vLLM imports.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from unittest.mock import patch

OPTION_DEFAULTS = {
    "VLLM_INDEXER_LOGITS_MAXNREG": 0,
    "VLLM_INDEXER_LOGITS_KV_GROUP_MIN_M": 0,
    "VLLM_INDEXER_LOGITS_FACTOR_K_SCALE": True,
}
SRC_REL = "v1/attention/ops/mqa_logits_triton.py"
HELPER_NAMES = ("_indexer_int_env", "_indexer_factor_k_scale")


def run(src: Path, upstream_envs: Path) -> None:
    source = src.read_text(encoding="utf-8")
    upstream = upstream_envs.read_text(encoding="utf-8")
    for option in OPTION_DEFAULTS:
        if option in upstream:
            raise AssertionError(
                f"{option} unexpectedly became upstream: recheck port behavior"
            )
        if option not in source:
            raise AssertionError(f"lost compatibility control {option}")

    tree = ast.parse(source, filename=str(src))
    attr_refs = [
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "envs"
    ]
    if attr_refs:
        raise AssertionError(f"vendored code accesses vllm.envs attributes: {attr_refs}")

    helpers = [
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in HELPER_NAMES
    ]
    if len(helpers) != len(HELPER_NAMES):
        raise AssertionError("all source-local tuning helper functions must exist")
    import_os = ast.Import(names=[ast.alias(name="os")])
    isolated = ast.fix_missing_locations(ast.Module(body=[import_os] + helpers, type_ignores=[]))
    globals_: dict = {}
    exec(compile(isolated, str(src), "exec"), globals_)
    get_int = globals_["_indexer_int_env"]
    get_scale = globals_["_indexer_factor_k_scale"]
    with patch.dict(os.environ, {}, clear=True):
        assert get_int("VLLM_INDEXER_LOGITS_MAXNREG") == 0
        assert get_int("VLLM_INDEXER_LOGITS_KV_GROUP_MIN_M") == 0
        assert get_scale() is True
    with patch.dict(os.environ, {
        "VLLM_INDEXER_LOGITS_MAXNREG": "128",
        "VLLM_INDEXER_LOGITS_KV_GROUP_MIN_M": "512",
        "VLLM_INDEXER_LOGITS_FACTOR_K_SCALE": "0",
    }, clear=True):
        assert get_int("VLLM_INDEXER_LOGITS_MAXNREG") == 128
        assert get_int("VLLM_INDEXER_LOGITS_KV_GROUP_MIN_M") == 512
        assert get_scale() is False
    with patch.dict(os.environ, {
        "VLLM_INDEXER_LOGITS_MAXNREG": "not-an-int"
    }, clear=True):
        try:
            get_int("VLLM_INDEXER_LOGITS_MAXNREG")
        except ValueError:
            pass
        else:
            raise AssertionError("invalid tuning integer silently accepted")
    for expected in [
        'maxnreg=_indexer_int_env("VLLM_INDEXER_LOGITS_MAXNREG") or None',
        'min_m = _indexer_int_env("VLLM_INDEXER_LOGITS_KV_GROUP_MIN_M")',
        "FACTOR_K_SCALE=_indexer_factor_k_scale()",
    ]:
        if expected not in source:
            raise AssertionError(f"missing source wiring: {expected}")
    print("SM80_MQA_VLLM0320_ENV_COMPAT_PASS (defaults, overrides, invalid setting)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True,
                   help="Root of the patched vLLM Python package")
    args = p.parse_args()
    run(args.source_root / SRC_REL, args.source_root / "envs.py")


if __name__ == "__main__":
    main()
