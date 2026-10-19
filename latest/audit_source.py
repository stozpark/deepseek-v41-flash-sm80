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
    if mode in ("port", "numeric", "indexer", "engram", "dspark"):
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
    if mode in ("numeric", "indexer", "engram", "dspark"):
        numeric_sources = [
            "models/deepseek_v41/common/ops/cache_utils.py",
            "models/deepseek_v41/common/ops/fused_compress_quant_cache.py",
            "models/deepseek_v41/common/ops/indexer_k_store.py",
            "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
            "v1/attention/ops/rocm_aiter_mla_sparse.py",
        ]
        for rel in numeric_sources:
            source = (root / rel).read_text()
            if ".to(tl.float8e4nv" in source:
                fail(f"SM80-forbidden direct E4M3 cast in {rel}")
            if "vllm.v1.attention.ops.fp8_sm80" not in source:
                fail(f"SM80 software FP8 path missing in {rel}")
        rope = (root / "v1/attention/ops/rocm_aiter_mla_sparse.py").read_text()
        if "return output.shape[0] if fused_inv_rope else 0" not in rope:
            fail("inverse RoPE return contract is not fixed")
        checks.append("stage2_sm80_fp8_numeric_source_paths")
    if mode in ("indexer", "engram", "dspark"):
        source=(root / "model_executor/layers/sparse_attn_indexer.py").read_text()
        for needed in (
            "def _sm80_indexer_fallback",
            "fp8_paged_mqa_logits_triton(",
            "fp8_mqa_logits_triton(",
            "SOFTWARE_FP8=_sm80_indexer_fallback()",
            "SM80 DSpark5 supports at most six target rows" if mode == "dspark" else "SM80 plain one-token decode only",
        ):
            if needed not in source:
                fail(f"SM80 indexer is missing {needed}")
        checks.append("stage3_deepgemm_free_sm80_indexer")
    if mode in ("engram", "dspark"):
        source=(root / "models/deepseek_v41/common/engram.py").read_text()
        for required in (
            "SOFTWARE_FP8",
            "_decode_fp8_f32(values, False)",
            "weight.to(tl.pointer_type(tl.uint8))",
            "def _engram_sm80_weight_bytes(",
            "weight.view(torch.uint8)",
            "weight = _engram_sm80_weight_bytes(weight)",
            "SOFTWARE_FP8=software_fp8",
        ):
            if required not in source:
                fail(f"Engram SM80 lookup source missing {required}")
        checks.append("stage4_engram_u8_argument_and_byte_dequant")
    if mode in ("port", "numeric", "indexer", "engram", "dspark"):
        mqa = (root / "v1/attention/ops/mqa_logits_triton.py").read_text()
        if "envs.VLLM_INDEXER_LOGITS_" in mqa:
            fail("old SM80 MQA port uses vllm.envs attributes removed in 0.32.0")
        for name in ("_indexer_int_env", "_indexer_factor_k_scale"):
            if name not in mqa:
                fail(f"missing SM80 MQA compatibility helper: {name}")
        checks.append("v0320_mqa_env_api_compat")
    if mode == "dspark":
        source=(root / "v1/attention/ops/mqa_logits_triton.py").read_text()
        idx=(root / "model_executor/layers/sparse_attn_indexer.py").read_text()
        for needle in ("context_lens_ptr + token_id", "q_offset = context_len - 1", "row_lens"):
            if needle not in source:
                fail(f"Missing DSpark5 per-query causal semantics: {needle}")
        if "SM80 DSpark5 supports at most six target rows" not in idx:
            fail("DSpark5 SM80 target indexer is not activated")
        checks.append("stage5_dspark5_per_query_causal_mask")
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
    p.add_argument("--mode", choices=("baseline", "port", "numeric", "indexer", "engram", "dspark"), default="baseline")
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
