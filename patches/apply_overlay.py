#!/usr/bin/env python3
from __future__ import annotations

import compileall
import shutil
from pathlib import Path

import vllm

SRC = Path("/opt/sm80-overlay/vllm")
DST = Path(vllm.__file__).resolve().parent

required_base = [
    DST / "models/deepseek_v4_1",
    DST / "model_executor/kernels/linear/mxfp4/marlin.py",
    DST / "model_executor/kernels/linear/mxfp8/emulation.py",
]
missing_base = [str(p) for p in required_base if not p.exists()]
if missing_base:
    raise SystemExit(
        "Official DeepSeek-V4.1 image is not compatible with this overlay; "
        f"missing baseline paths: {missing_base}"
    )

if not SRC.is_dir():
    raise SystemExit(f"SM80 overlay missing: {SRC}")

copied = 0
for src in sorted(SRC.rglob("*")):
    if not src.is_file():
        continue
    rel = src.relative_to(SRC)
    dst = DST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(
            f"{description}: expected baseline text not found in {path}; "
            "refusing an unsafe patch"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Dependency closure for the SM80 sparse-indexer query/decode sharding added
# by wtdcode/vllm-backport a6ef07a + 6793ead. The official 0909 image predates
# these two shared helpers, while the pinned SM80 indexer imports them.
# Keep the official distributed/utils.py otherwise untouched.
dist_utils = DST / "distributed/utils.py"
dist_text = dist_utils.read_text(encoding="utf-8")
if "def balanced_row_counts(" not in dist_text:
    dist_text += '''\n\n# SM80 backport compatibility: shared balanced TP row partition.\ndef balanced_row_counts(num_rows: int, size: int) -> list[int]:\n    """Per-rank row counts using base + (rank < remainder)."""\n    base, rem = divmod(num_rows, size)\n    return [base + (rank < rem) for rank in range(size)]\n\n\ndef balanced_row_bounds(\n    start: int, stop: int, rank: int, size: int\n) -> tuple[int, int]:\n    """This rank's contiguous half-open balanced range in [start, stop)."""\n    base, rem = divmod(stop - start, size)\n    lo = start + rank * base + min(rank, rem)\n    return lo, lo + base + (rank < rem)\n'''
    dist_utils.write_text(dist_text, encoding="utf-8")
elif "def balanced_row_bounds(" not in dist_text:
    raise SystemExit(
        "distributed.utils has balanced_row_counts but not balanced_row_bounds; "
        "unexpected baseline, refusing unsafe patch"
    )

# Post-0909 correctness hotfix: vllm-project/vllm#56297 / backport #78.
# Responses API uses input_text/output_text while the 0909 V4.1 tokenizer
# accepts only `text`.
tokenizer = DST / "tokenizers/deepseek_v41.py"
replace_once(
    tokenizer,
    'if part_type == "text":\n                    parts.append(block.get("text", ""))',
    'if part_type in ("text", "input_text", "output_text"):\n'
    '                    parts.append(block.get("text", ""))',
    "DeepSeek-V4.1 Responses API text-content hotfix",
)

# Post-0909 fix corresponding to vLLM PR #56623. A V4.1 vision-capable
# checkpoint served with --language-model-only must not reserve the extra
# vision-visible SWA width (1024 tokens for V4.1-Flash). This is useful on
# SM80 too: it avoids unnecessary metadata/index bandwidth. If multimodal
# config is absent or language_model_only is false, behavior is unchanged.
sparse_swa = DST / "v1/attention/backends/mla/sparse_swa.py"
replace_once(
    sparse_swa,
    '''        self.max_image_tokens = (\n            getattr(hf_config, "vision_max_n_token", 0)\n            if getattr(hf_config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
    '''        mm_config = getattr(self.vllm_config.model_config, "multimodal_config", None)\n        language_model_only = bool(getattr(mm_config, "language_model_only", False))\n        self.max_image_tokens = (\n            0\n            if language_model_only\n            else (\n                getattr(hf_config, "vision_max_n_token", 0)\n                if getattr(hf_config, "vision_n_layers", 0) > 0\n                else 0\n            )\n        )''',
    "DeepSeek-V4.1 language-model-only SWA-width hotfix",
)

v41_attention = DST / "models/deepseek_v4_1/attention.py"
replace_once(
    v41_attention,
    '''        self.max_image_tokens = (\n            getattr(config, "vision_max_n_token", 0)\n            if getattr(config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
    '''        mm_config = getattr(vllm_config.model_config, "multimodal_config", None)\n        language_model_only = bool(getattr(mm_config, "language_model_only", False))\n        self.max_image_tokens = (\n            0\n            if language_model_only\n            else (\n                getattr(config, "vision_max_n_token", 0)\n                if getattr(config, "vision_n_layers", 0) > 0\n                else 0\n            )\n        )''',
    "DeepSeek-V4.1 attention language-model-only SWA-width hotfix",
)

# SM80 mHC correctness: the 0909 branch's first-layer broadcast path invokes
# DeepGEMM unconditionally, but DeepGEMM's mHC kernel is Hopper+ only. The
# backport fix (later upstreamed as vLLM #50645) uses the already-present
# TileLang prenorm GEMM when DeepGEMM is unsupported.
mhc_tilelang = DST / "model_executor/kernels/mhc/tilelang.py"
replace_once(
    mhc_tilelang,
    '''    n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))''',
    '''    from vllm.utils.deep_gemm import is_deep_gemm_supported\n\n    use_deep_gemm = is_deep_gemm_supported()\n    if use_deep_gemm:\n        n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))\n    else:\n        n_splits = 1''',
    "SM80 mHC broadcast DeepGEMM guard",
)
replace_once(
    mhc_tilelang,
    '''    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm\n\n    tf32_hc_prenorm_gemm(\n        residual_flat,\n        fn_broadcast,\n        gemm_out_mul,\n        gemm_out_sqrsum,\n        n_splits,\n    )''',
    '''    if use_deep_gemm:\n        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm\n\n        tf32_hc_prenorm_gemm(\n            residual_flat,\n            fn_broadcast,\n            gemm_out_mul,\n            gemm_out_sqrsum,\n            n_splits,\n        )\n    else:\n        _tilelang_hc_prenorm_gemm(\n            residual_flat,\n            fn_broadcast,\n            gemm_out_mul,\n            gemm_out_sqrsum,\n            hidden_size,\n            1,\n        )''',
    "SM80 mHC broadcast TileLang fallback",
)

# vLLM #53376: CUTLASS FP8 auto-selection must decline SM80. Without this,
# mixed FP8 linear layers can choose a CUTLASS kernel that has no Ampere
# implementation instead of falling through to the intended Marlin fallback.
cutlass_fp8 = DST / "model_executor/kernels/linear/scaled_mm/cutlass.py"
replace_once(
    cutlass_fp8,
    '''        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        return True, None''',
    '''        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        if compute_capability is None:\n            capability_tuple = current_platform.get_device_capability()\n            compute_capability = (\n                -1 if capability_tuple is None else capability_tuple.to_int()\n            )\n        if not ops.cutlass_scaled_mm_supports_fp8(compute_capability):\n            return (\n                False,\n                "CUTLASS FP8 GEMM is unavailable for compute capability "\n                f"{compute_capability} with this CUDA build (needs SM89 with "\n                "CUDA 12.4 or newer, or SM90+ with CUDA 12.0 or newer).",\n            )\n        return True, None''',
    "SM80 CUTLASS FP8 capability gate",
)

# vLLM #55109: narrowed block-table views retain the backing tensor's larger
# row stride. The pinned SM80 gather kernel used shape[-1], so request 2+
# could read the wrong physical block and issue an OOB access. V4 and V4.1
# carry separate copies of this kernel in the pinned tree; fix both.
for cache_utils in (
    DST / "models/deepseek_v4/common/ops/cache_utils.py",
    DST / "models/deepseek_v4_1/common/ops/cache_utils.py",
):
    replace_once(
        cache_utils,
        "    max_blocks_per_seq: tl.constexpr,\n",
        "    block_table_stride: tl.constexpr,\n",
        "K-cache gather physical block-table stride parameter",
    )
    replace_once(
        cache_utils,
        "block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq",
        "block_table_row_ptr = block_table_ptr + batch_idx * block_table_stride",
        "K-cache gather physical block-table row addressing",
    )
    replace_once(
        cache_utils,
        "        max_blocks_per_seq=block_table.shape[-1],\n",
        "        block_table_stride=block_table.stride(0),\n",
        "K-cache gather physical block-table stride launch",
    )

required_overlay = [
    DST / "v1/attention/ops/fp8_sm80.py",
    DST / "v1/attention/ops/mqa_logits_triton.py",
    DST / "models/deepseek_v4_1/ampere/ampere_sparse.py",
    DST / "v1/attention/backends/mla/indexer.py",
]
missing_overlay = [str(p) for p in required_overlay if not p.exists()]
if missing_overlay:
    raise SystemExit(f"SM80 overlay incomplete after copy: {missing_overlay}")

# Long-context correctness invariants from the SM80 paged-indexer fallback.
mqa_text = (DST / "v1/attention/ops/mqa_logits_triton.py").read_text(encoding="utf-8")
for required in (".to(tl.int64)", "k_offset < context_len"):
    if required not in mqa_text:
        raise SystemExit(
            "SM80 paged MQA logits is missing the long-context correctness fix: "
            + required
        )

# The pinned backport already contains the fourth #50576 correctness fix:
# fused inverse-RoPE passes launch_pdl in both the kernel dispatcher and wrapper.
inv_rope = (DST / "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py").read_text(
    encoding="utf-8"
)
if inv_rope.count("launch_pdl=launch_pdl") < 2:
    raise SystemExit("fused_inv_rope_fp8_quant is missing the launch_pdl fix")

if not compileall.compile_dir(str(DST), quiet=1, force=False):
    raise SystemExit("Python compile check failed after SM80 overlay")

marker = Path("/opt/sm80-overlay/APPLIED")
marker.write_text(
    f"target={DST}\nfiles={copied}\n"
    "balanced_row_helpers=1\n"
    "responses_api_text_hotfix=1\n"
    "language_model_only_swa_hotfix=1\n"
    "mhc_sm80_fallback=1\n"
    "cutlass_fp8_sm80_gate=1\n"
    "strided_block_table_gather=1\n"
    "mqa_long_context_guards=1\n"
    "launch_pdl_fix_present=1\n",
    encoding="utf-8",
)
print(f"Applied {copied} SM80 overlay files to {DST}")
print("Applied SM80 dependency-closure and post-0909 guarded hotfixes")
