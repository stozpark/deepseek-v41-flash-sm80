#!/usr/bin/env python3
"""Static vLLM v0.32.0 / DeepSeek V4.1 SM80 compatibility audit.

This deliberately does NOT claim that passing syntax/source checks proves
physical A100 kernel correctness or end-to-end inference.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

EXPECTED_VERSION = "0.32.0"
MODEL_PACKAGE = "models/deepseek_v41"
REGISTRY = "v1/attention/backends/registry.py"


def fail(message: str) -> None:
    raise SystemExit("FAIL: " + message)


def must_include(root: Path, relative: str, needles: tuple[str, ...]) -> None:
    target = root / relative
    if not target.is_file():
        fail(f"missing file {relative}")
    src = target.read_text(encoding="utf-8")
    for needle in needles:
        if needle not in src:
            fail(f"missing {needle!r} in {relative}")


def check_source(root: Path, mode: str) -> list[str]:
    checks: list[str] = []
    must_include(
        root,
        f"{MODEL_PACKAGE}/nvidia/model.py",
        ("def _select_dsv4_attn_cls(", "DeepseekV4FlashMLAAttention"),
    )
    checks.append("upstream_v41_model")
    must_include(
        root, f"{MODEL_PACKAGE}/attention.py",
        ("class DeepseekV4Attention", "candidate_topk_blocks"),
    )
    checks.append("two_level_indexer")
    must_include(
        root, f"{MODEL_PACKAGE}/common/engram.py",
        ("Engram",),
    )
    checks.append("engram")
    must_include(
        root, REGISTRY,
        ("FLASHMLA_SPARSE_DSV41", "FLASHINFER_MLA_SPARSE_DSV41"),
    )
    checks.append("v41_backend_registry")
    if (root / "models/deepseek_v4_1").exists():
        fail("old deepseek_v4_1 package found: not the clean v0.32.0 baseline")
    checks.append("renamed_v41_package")
    if mode == "port":
        must_include(
            root,
            f"{MODEL_PACKAGE}/ampere/ampere_sparse.py",
            ("DeepseekV41AmpereMLASparseBackend", "DeepseekV41AmpereMLAAttention"),
        )
        must_include(
            root, f"{MODEL_PACKAGE}/nvidia/model.py",
            ("DeepseekV41AmpereMLAAttention", "device_capability.major == 8"),
        )
        must_include(root, REGISTRY, ("TRITON_MLA_SPARSE_DSV41",))
        must_include(root, "v1/attention/ops/fp8_sm80.py", ("get_e4m3fn_bf16_lut",))
        must_include(root, "v1/attention/ops/mqa_logits_triton.py", ("fp8_paged_mqa_logits_triton",))
        checks.append("stage1_sm80_backend_and_kernel_files")
    return checks


def runtime_audit() -> dict:
    try:
        import torch
    except ImportError as exc:
        fail(f"torch is not installed: {exc}")
    cuda = str(torch.version.cuda or "")
    try:
        major = int(cuda.split(".")[0])
    except (ValueError, IndexError):
        fail(f"CUDA runtime is missing or invalid: {cuda!r}")
    if major < 13:
        fail(f"CUDA >= 13 required, got {cuda}")
    if not torch.cuda.is_available():
        fail("no GPU available; runtime test requires --nv and A100 access")
    gpus = []
    for idx in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(idx)
        name = torch.cuda.get_device_name(idx)
        if tuple(cap) != (8, 0) or "A100" not in name and "A800" not in name:
            fail(f"GPU {idx} is not A100/A800 SM80: {name}, {cap}")
        gpus.append(dict(id=idx, name=name, sm="80"))
    if len(gpus) < 8:
        fail(f"at least eight visible A100 GPUs required, got {len(gpus)}")
    return dict(cuda=cuda, torch=torch.__version__, visible_gpus=gpus)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("baseline", "port"), default="baseline")
    p.add_argument("--source-root", type=Path, help="vllm Python package directory")
    p.add_argument("--runtime", action="store_true", help="require CUDA>=13 and A100")
    p.add_argument("--model-config", type=Path, help="optional DeepSeek 1017 config.json")
    args = p.parse_args()
    if args.source_root is None:
        try:
            import vllm
        except ImportError as exc:
            fail(f"vllm is not installed: {exc}")
        version = importlib.metadata.version("vllm")
        if version != EXPECTED_VERSION:
            fail(f"pin mismatch: expected vLLM {EXPECTED_VERSION}, got {version}")
        root = Path(vllm.__file__).resolve().parent
    else:
        root = args.source_root.resolve()
        version = "source-tree (version verified separately)"
    checks = check_source(root, args.mode)
    extra = {}
    if args.model_config:
        cfg = json.loads(args.model_config.read_text(encoding="utf-8"))
        text = cfg.get("text_config", cfg)
        if cfg.get("architectures") != ["DeepseekV41ForCausalLM"]:
            fail("unexpected model architecture")
        o_groups = int(text.get("o_groups", -1))
        if o_groups != 8 or int(text.get("num_attention_heads", -1)) != 64:
            fail("1017 model topology changed: revalidate TP8 before serving")
        extra = dict(o_groups=o_groups, num_attention_heads=64, max_supported_tp=o_groups)
        checks.append("1017_model_topology")
    if args.runtime:
        extra.update(runtime_audit())
        checks.append("runtime_sm80_cuda13")
    print(json.dumps(dict(
        result="SOURCE_AUDIT_PASS_NOT_INFERENCE_VALIDATION",
        mode=args.mode, vllm_version=version, root=str(root), checks=checks, **extra
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
