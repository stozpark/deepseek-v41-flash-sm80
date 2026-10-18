#!/usr/bin/env python3
"""Stage3 SM80 accuracy-first DeepGEMM-free sparse indexer.

This enables plain FP8 one-token sparse Q/K scoring with BF16 Triton MQA
instead of unsupported FP8 DeepGEMM. MXFP4 indexer, PCP, DCP, spec decode
must fail closed. GPU numeric validation is mandatory before model serving.
"""
from __future__ import annotations
import argparse
import compileall
import hashlib
import importlib.metadata
import re
import shutil
from pathlib import Path

VERSION = "0.32.0"
RELATIVE = "model_executor/layers/sparse_attn_indexer.py"
ORIGINAL = "935e4c0d2d446059c620e4729bbbadb79e76ca64"


def oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}".encode()+bytes([0])+data).hexdigest()


def change(src: str, old: str, new: str, expected: int=1) -> str:
    count=src.count(old)
    if count!=expected:
        raise RuntimeError(f"Expected {expected} anchors, saw {count}: {old[:140]!r}")
    return src.replace(old,new)


def create_patch(root: Path) -> tuple[Path, str]:
    f=root/RELATIVE
    data=f.read_bytes()
    if oid(data)!=ORIGINAL:
        raise RuntimeError(f"Refusing unknown indexer source {oid(data)} != {ORIGINAL}")
    s=data.decode()
    s=change(s,
        "from vllm.triton_utils import tl, triton\n",
        "from vllm.triton_utils import tl, triton\n"
        "from vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8\n")
    s=change(s,
        "# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.\n",
        '''def _sm80_indexer_fallback() -> bool:
    return current_platform.is_cuda() and current_platform.is_device_capability_family(80)


@triton.jit
def _sm80_indexer_quant_value(x, SOFTWARE_FP8: tl.constexpr, DTYPE: tl.constexpr):
    if SOFTWARE_FP8:
        return _encode_e4m3fn_u8(x)
    return x.to(DTYPE)


# MXFP4 layout: 2 values packed per byte, ue8m0 (1-byte) scale per block of 32.
''')
    s=change(s,
        "    is_neox: tl.constexpr,\n):\n    token = tl.program_id(0)",
        "    is_neox: tl.constexpr,\n    SOFTWARE_FP8: tl.constexpr,\n):\n    token = tl.program_id(0)")
    pattern=r"tl\.clamp\((r0|r1|q_nope) / q_scale, fp8_min, fp8_max\)\.to\(q_fp8\.dtype\.element_ty\)"
    matches=re.findall(pattern,s)
    if len(matches)!=5:
        raise RuntimeError(f"Expected five fp8 q stores, got {len(matches)}")
    s=re.sub(pattern,lambda m:
        "_sm80_indexer_quant_value(tl.clamp("+m.group(1)+
        " / q_scale, fp8_min, fp8_max), SOFTWARE_FP8, q_fp8.dtype.element_ty)",
        s)
    s=change(s,"        q_fp8,\n        q_fp8.stride(0),",
        "        q_fp8.view(torch.uint8) if _sm80_indexer_fallback() else q_fp8,\n"
        "        q_fp8.stride(0),")
    s=change(s,"        is_neox=is_neox,\n        num_warps=1,",
        "        is_neox=is_neox,\n        SOFTWARE_FP8=_sm80_indexer_fallback(),\n        num_warps=1,")
    s=change(s,
        '''                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )''',
        '''                elif _sm80_indexer_fallback():
                    if use_fp4_cache or q_scale_slice is not None:
                        raise RuntimeError("SM80 v4.1 requires FP8 indexer K/Q")
                    if use_pcp or dcp_world_size != 1:
                        raise RuntimeError("SM80 v4.1 indexer cannot do PCP/DCP")
                    from vllm.v1.attention.ops.mqa_logits_triton import (
                        fp8_mqa_logits_triton,
                    )
                    logits = fp8_mqa_logits_triton(
                        q_slice_cast,
                        (k_quant_cast, k_scale_cast),
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )
                else:
                    logits = fp8_fp4_mqa_logits(
                        (q_slice_cast, q_scale_slice),
                        (k_quant_cast, k_scale_cast),
                        weights[row_start:row_end],
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        clean_logits=False,
                    )''')
    s=change(s,
        '''        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )''',
        '''        elif _sm80_indexer_fallback():
            if use_fp4_cache or padded_q_scale is not None:
                raise RuntimeError("SM80 paged MQA requires FP8 K/Q")
            if next_n != 1 or dcp_world_size != 1 or use_pcp:
                raise RuntimeError("SM80 plain one-token decode only (no DCP/PCP/spec)")
            from vllm.v1.attention.ops.mqa_logits_triton import (
                fp8_paged_mqa_logits_triton,
            )
            logits = fp8_paged_mqa_logits_triton(
                padded_q_quant_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens.reshape(-1).contiguous(),
                decode_metadata.block_table,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        else:
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
                indices=decode_metadata.indices,
            )''')
    s=change(s,
        "if current_platform.is_cuda() and not has_deep_gemm():\n            raise RuntimeError(",
        "if current_platform.is_cuda() and not has_deep_gemm() and not _sm80_indexer_fallback():\n"
        "            raise RuntimeError(")
    s=change(s,
        "        self._cp_kv_cache_interleave_size: int | None = None\n",
        '''        self._cp_kv_cache_interleave_size: int | None = None
        if _sm80_indexer_fallback():
            if self.use_fp4_cache or self.dcp_world_size != 1 or self.use_pcp:
                raise RuntimeError("SM80 indexer requires FP8 and no DCP/PCP")
''')
    compile(s,str(f),"exec")
    return f,s


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--skip-version-check",action="store_true")
    args=parser.parse_args()
    if not args.skip_version_check and importlib.metadata.version("vllm")!=VERSION:
        raise RuntimeError("vLLM version mismatch")
    if args.source_root:
        root=args.source_root.resolve()
    else:
        import vllm
        root=Path(vllm.__file__).resolve().parent
    dst,new=create_patch(root)
    if args.dry_run:
        print("STAGE3_SM80_INDEXER_PATCH_PLAN_PASS")
        return
    backup=dst.with_name(dst.name+".sm80-v0320.original")
    if backup.exists():
        raise RuntimeError("Refusing repeated stage3 patch")
    shutil.copy2(dst,backup)
    try:
        dst.write_text(new)
        if not compileall.compile_file(str(dst),quiet=1,force=True):
            raise RuntimeError("Patched indexer syntax check failed")
    except Exception:
        shutil.move(str(backup),str(dst))
        raise
    print("STAGE3_SM80_INDEXER_SOURCE_PASS_NOT_GPU_VALIDATED")

if __name__=="__main__":
    main()
