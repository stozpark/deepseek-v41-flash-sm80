#!/usr/bin/env python3
"""Build a production-minimal DeepSeek-V4.1 SM80 text tree.

Inputs are text snapshots of (1) the untouched official 0909-cu129 vLLM and
(2) the already replay-verified broad SM80 backport. We deliberately copy only
the files that form the DeepSeek-V4.1 / A100 dependency closure. For shared
files that also contain unrelated post-0909 work, we start from the official
baseline and apply only the specific SM80/V4.1 hunks we need.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


FULL_COPY = (
    # shared compatibility / capability guards
    "distributed/utils.py",
    "model_executor/kernels/linear/scaled_mm/cutlass.py",
    "model_executor/kernels/mhc/tilelang.py",
    "model_executor/layers/sparse_attn_indexer.py",
    "model_executor/warmup/cutedsl_warmup.py",
    "model_executor/warmup/flashinfer_sparse_mla_warmup.py",
    "utils/import_utils.py",
    # V4 modules explicitly reused by V4.1 common/ops
    "models/deepseek_v4/common/ops/fused_indexer_q.py",
    "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py",
    # V4.1 SM80 model/cache path
    "models/deepseek_v4_1/ampere/__init__.py",
    "models/deepseek_v4_1/ampere/ampere_sparse.py",
    "models/deepseek_v4_1/attention.py",
    "models/deepseek_v4_1/common/ops/cache_utils.py",
    "models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py",
    "models/deepseek_v4_1/common/ops/indexer_k_store.py",
    # API correctness
    "tokenizers/deepseek_v41.py",
    # sparse indexer / portable SM80 kernels
    "v1/attention/backends/mla/indexer.py",
    "v1/attention/ops/common.py",
    "v1/attention/ops/fp8_sm80.py",
    "v1/attention/ops/mqa_logits_triton.py",
    "v1/attention/ops/rocm_aiter_mla_sparse.py",
)


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"{label}: expected exactly one baseline match in {path}, got {count}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_v41_nvidia_model(root: Path) -> None:
    p = root / "models/deepseek_v4_1/nvidia/model.py"
    anchor = "    device_capability = current_platform.get_device_capability()\n"
    addition = anchor + '''    if device_capability is not None and device_capability.major == 8:\n        if backend is not None and (\n            backend != AttentionBackendEnum.TRITON_MLA_SPARSE_DSV41\n        ):\n            raise ValueError(\n                f"{backend.name} is not supported for DeepSeek V4.1 on SM8x; "\n                "use TRITON_MLA_SPARSE_DSV41 (default)."\n            )\n        # indexer_kv_dtype="mxfp4" is rejected by dsa_indexer_uses_fp4().\n        from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (\n            DeepseekV41AmpereMLAAttention,\n        )\n\n        return DeepseekV41AmpereMLAAttention\n'''
    replace_once(p, anchor, addition, "V4.1 SM80 attention routing")


def patch_v41_engram(root: Path) -> None:
    p = root / "models/deepseek_v4_1/common/engram.py"

    # Keep every official 0909 DP/ubatching feature. Only make the lookup
    # representation pre-SM89-safe by passing raw bytes and decoding in Triton.
    anchor = "from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor\n"
    addition = anchor + "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32\n"
    replace_once(p, anchor, addition, "V4.1 Engram SM80 FP8 helper import")

    old_load = '''        values = tl.load(\n            weight + local[:, None] * DIM + cols[None, :],\n            mask=owned[:, None],\n            other=0.0,\n        )\n        scale = tl.load('''
    new_load = '''        values = tl.load(\n            weight + local[:, None] * DIM + cols[None, :],\n            mask=owned[:, None],\n            other=0,\n        )\n        values = _decode_fp8_f32(values, False)\n        scale = tl.load('''
    replace_once(p, old_load, new_load, "V4.1 Engram SM80 FP8 decode")

    replace_once(
        p,
        '(values.to(tl.float32) * scale).to(tl.bfloat16)',
        '(values * scale).to(tl.bfloat16)',
        "V4.1 Engram decoded-value multiply",
    )

    old_call = '''        _engram_lookup_kernel[(grid,)](\n            weight,\n            scales,'''
    new_call = '''        _engram_lookup_kernel[(grid,)](\n            weight.view(torch.uint8),\n            scales,'''
    replace_once(p, old_call, new_call, "V4.1 Engram raw-byte kernel argument")


def patch_registry(root: Path) -> None:
    p = root / "v1/attention/backends/registry.py"
    anchor = '''    FLASHINFER_MLA_SPARSE_DSV41 = (\n        "vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse."\n        "DeepseekV4FlashInferMLASparseBackend"\n    )\n'''
    addition = anchor + '''    TRITON_MLA_SPARSE_DSV41 = (\n        "vllm.models.deepseek_v4_1.ampere.ampere_sparse."\n        "DeepseekV41AmpereMLASparseBackend"\n    )\n'''
    replace_once(p, anchor, addition, "V4.1 SM80 backend registry")


def patch_sparse_swa(root: Path) -> None:
    p = root / "v1/attention/backends/mla/sparse_swa.py"
    text = p.read_text(encoding="utf-8")
    if "def swa_max_image_tokens(" not in text:
        class_anchor = "\n\nclass DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):"
        helper = '''\n\ndef swa_max_image_tokens(vllm_config: VllmConfig) -> int:\n    """Extra SWA width needed only when images can actually reach the model."""\n    hf_config = vllm_config.model_config.hf_config\n    if getattr(hf_config, "vision_n_layers", 0) <= 0:\n        return 0\n    mm_config = vllm_config.model_config.multimodal_config\n    if mm_config is not None and mm_config.language_model_only:\n        return 0\n    return int(getattr(hf_config, "vision_max_n_token", 0) or 0)\n'''
        if text.count(class_anchor) != 1:
            raise SystemExit("V4.1 SWA helper: class anchor mismatch")
        p.write_text(text.replace(class_anchor, helper + class_anchor, 1), encoding="utf-8")

    replace_once(
        p,
        "        if current_platform.is_rocm():\n",
        '''        if current_platform.is_rocm() or (\n            current_platform.is_cuda()\n            and not current_platform.has_device_capability(90)\n        ):\n''',
        "V4.1 SM80 ragged SWA metadata builder",
    )
    replace_once(
        p,
        '''        self.max_image_tokens = (\n            getattr(hf_config, "vision_max_n_token", 0)\n            if getattr(hf_config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
        "        self.max_image_tokens = swa_max_image_tokens(self.vllm_config)",
        "V4.1 text-only SWA width",
    )
    replace_once(
        p,
        '''            or current_platform.is_device_capability_family(120)\n        ):''',
        '''            or current_platform.is_device_capability_family(120)\n            # CUDA SM8x shares the ROCm ragged Triton decode kernels.\n            or (\n                current_platform.is_cuda()\n                and not current_platform.has_device_capability(90)\n            )\n        ):''',
        "V4.1 SM80 SWA scheduler guard",
    )


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: minimize_patched_tree.py BASELINE FULL_PATCHED OUTPUT_MINIMAL"
        )
    baseline = Path(sys.argv[1]).resolve()
    full = Path(sys.argv[2]).resolve()
    out = Path(sys.argv[3]).resolve()
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(baseline, out)

    for rel in FULL_COPY:
        src = full / rel
        if not src.is_file():
            raise SystemExit(f"validated full tree is missing required file: {rel}")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    patch_v41_nvidia_model(out)
    patch_v41_engram(out)
    patch_registry(out)
    patch_sparse_swa(out)

    print("minimal_full_copy_files=", len(FULL_COPY))
    print("minimal_targeted_files=4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
