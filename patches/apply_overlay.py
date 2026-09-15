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


def replace_in_class(
    path: Path,
    class_name: str,
    old: str,
    new: str,
    description: str,
) -> None:
    """Replace only inside one top-level class, never a similar sibling class."""
    text = path.read_text(encoding="utf-8")
    marker = f"class {class_name}"
    start = text.find(marker)
    if start < 0:
        raise SystemExit(f"{description}: class {class_name} not found in {path}")
    end = text.find("\nclass ", start + len(marker))
    if end < 0:
        end = len(text)
    body = text[start:end]
    if new in body:
        return
    if old not in body:
        raise SystemExit(
            f"{description}: expected text not found inside {class_name} in {path}; "
            "refusing an unsafe patch"
        )
    body = body.replace(old, new, 1)
    path.write_text(text[:start] + body + text[end:], encoding="utf-8")


# Dependency closure for the SM80 sparse-indexer query/decode sharding added
# by wtdcode/vllm-backport a6ef07a + 6793ead. The official 0909 image predates
# these shared helpers, while the pinned SM80 indexer imports them.
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

# The pinned SM80 DeepSeek files use this gate to avoid compiling CuTe DSL
# kernels below Hopper. The 0909 image has has_cutedsl() but predates the
# capability-aware helper from the SM80 backport.
import_utils = DST / "utils/import_utils.py"
import_text = import_utils.read_text(encoding="utf-8")
if "def is_cutedsl_supported(" not in import_text:
    anchor = '''def has_cutedsl() -> bool:\n    """Whether the optional `cutelass` package is available."""\n    return _has_module("cutlass")\n'''
    addition = anchor + '''\n\n@cache\ndef is_cutedsl_supported() -> bool:\n    """Whether CuTe DSL is installed and can compile for this device."""\n    from vllm.platforms import current_platform\n\n    return has_cutedsl() and current_platform.has_device_capability(90)\n'''
    if anchor not in import_text:
        raise SystemExit(
            "CuTe DSL compatibility helper: expected has_cutedsl baseline not found; "
            "refusing an unsafe patch"
        )
    import_utils.write_text(import_text.replace(anchor, addition, 1), encoding="utf-8")

# Post-0909 correctness hotfix: vllm-project/vllm#56297 / backport #78.
tokenizer = DST / "tokenizers/deepseek_v41.py"
replace_once(
    tokenizer,
    'if part_type == "text":\n                    parts.append(block.get("text", ""))',
    'if part_type in ("text", "input_text", "output_text"):\n'
    '                    parts.append(block.get("text", ""))',
    "DeepSeek-V4.1 Responses API text-content hotfix",
)

# vLLM #56623: a DeepSeek V4.1 vision checkpoint served with
# --language-model-only must keep the plain 128-token SWA width. Mirror the
# upstream helper and apply it at the V4.1 attention, metadata-builder, and
# warmup-key call sites. This avoids warming/allocating the unused 1152-wide
# image-visible path on text-only A100 deployments.
sparse_swa = DST / "v1/attention/backends/mla/sparse_swa.py"
sparse_text = sparse_swa.read_text(encoding="utf-8")
if "def swa_max_image_tokens(" not in sparse_text:
    anchor = "\n\nclass DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):"
    helper = '''\n\ndef swa_max_image_tokens(vllm_config: VllmConfig) -> int:\n    """Extra SWA width needed only when images can actually reach the model."""\n    hf_config = vllm_config.model_config.hf_config\n    if getattr(hf_config, "vision_n_layers", 0) <= 0:\n        return 0\n    mm_config = vllm_config.model_config.multimodal_config\n    if mm_config is not None and mm_config.language_model_only:\n        return 0\n    return int(getattr(hf_config, "vision_max_n_token", 0) or 0)\n'''
    if anchor not in sparse_text:
        raise SystemExit(
            "DeepSeek-V4.1 language-model-only SWA helper: class anchor not found; "
            "refusing an unsafe patch"
        )
    sparse_swa.write_text(
        sparse_text.replace(anchor, helper + anchor, 1), encoding="utf-8"
    )

replace_once(
    sparse_swa,
    '''        self.max_image_tokens = (\n            getattr(hf_config, "vision_max_n_token", 0)\n            if getattr(hf_config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
    '''        self.max_image_tokens = swa_max_image_tokens(self.vllm_config)''',
    "DeepSeek-V4.1 language-model-only sparse-SWA width hotfix",
)

v41_attention = DST / "models/deepseek_v4_1/attention.py"
replace_once(
    v41_attention,
    "from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache",
    '''from vllm.v1.attention.backends.mla.sparse_swa import (\n    DeepseekV4SWACache,\n    swa_max_image_tokens,\n)''',
    "DeepSeek-V4.1 SWA helper import",
)
replace_once(
    v41_attention,
    '''        self.max_image_tokens = (\n            getattr(config, "vision_max_n_token", 0)\n            if getattr(config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
    '''        self.max_image_tokens = swa_max_image_tokens(vllm_config)''',
    "DeepSeek-V4.1 attention language-model-only SWA-width hotfix",
)

v41_cache_utils = DST / "models/deepseek_v4_1/common/ops/cache_utils.py"
replace_once(
    v41_cache_utils,
    '''        image_width = (\n            _hf_config_int(vllm_config, "vision_max_n_token", 0)\n            if _hf_config_int(vllm_config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
    '''        from vllm.v1.attention.backends.mla.sparse_swa import swa_max_image_tokens\n\n        image_width = swa_max_image_tokens(vllm_config)''',
    "DeepSeek-V4.1 warmup language-model-only SWA-width hotfix",
)

# vLLM #50645 / SM8x mHC: DeepGEMM's mHC kernel is Hopper+ only.
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

# vLLM #53376: CUTLASS FP8 auto-selection must decline SM80 and fall through
# to the Marlin fallback. Scope this replacement to the FP8 class only.
cutlass_fp8 = DST / "model_executor/kernels/linear/scaled_mm/cutlass.py"
replace_in_class(
    cutlass_fp8,
    "CutlassFP8ScaledMMLinearKernel",
    '''        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        return True, None''',
    '''        if not current_platform.is_cuda():\n            return False, "requires CUDA."\n        if compute_capability is None:\n            capability_tuple = current_platform.get_device_capability()\n            compute_capability = (\n                -1 if capability_tuple is None else capability_tuple.to_int()\n            )\n        if not ops.cutlass_scaled_mm_supports_fp8(compute_capability):\n            return (\n                False,\n                "CUTLASS FP8 GEMM is unavailable for compute capability "\n                f"{compute_capability} with this CUDA build (needs SM89 with "\n                "CUDA 12.4 or newer, or SM90+ with CUDA 12.0 or newer).",\n            )\n        return True, None''',
    "SM80 CUTLASS FP8 capability gate",
)

# vLLM #55109: narrowed block-table views must use the physical row stride.
# V4.1 has its own cache utilities; V4 is also patched because the V4.1 model
# imports shared V4 NVIDIA model/projection code during class construction.
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
    DST / "model_executor/kernels/linear/gemv_triton.py",
    DST / "v1/attention/ops/fp8_sm80.py",
    DST / "v1/attention/ops/mqa_logits_triton.py",
    DST / "models/deepseek_v4_1/ampere/ampere_sparse.py",
    DST / "v1/attention/backends/mla/indexer.py",
]
missing_overlay = [str(p) for p in required_overlay if not p.exists()]
if missing_overlay:
    raise SystemExit(f"SM80 overlay incomplete after copy: {missing_overlay}")

mqa_text = (DST / "v1/attention/ops/mqa_logits_triton.py").read_text(encoding="utf-8")
for required in (".to(tl.int64)", "k_offset < context_len"):
    if required not in mqa_text:
        raise SystemExit(
            "SM80 paged MQA logits is missing the long-context correctness fix: "
            + required
        )

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
    "cutedsl_capability_gate=1\n"
    "responses_api_text_hotfix=1\n"
    "language_model_only_swa_hotfix=1\n"
    "language_model_only_warmup_hotfix=1\n"
    "mhc_sm80_fallback=1\n"
    "cutlass_fp8_sm80_gate=1\n"
    "strided_block_table_gather=1\n"
    "mqa_long_context_guards=1\n"
    "launch_pdl_fix_present=1\n",
    encoding="utf-8",
)
print(f"Applied {copied} SM80 overlay files to {DST}")
print("Applied DeepSeek-V4.1 SM80 dependency closure and guarded hotfixes")
