#!/usr/bin/env python3
"""Stage4: SM80 Engram host-table FP8 exact byte interpretation.

The original DeepSeek V4.1 Triton lookup directly loads FP8 weights and
converts them to FP32. On A100 this is a native SM89+ E4M3 conversion, so
reinterpret as u8 and use bit-exact software decoder on SM80. The ordinary
(non-SM80) implementation is left unchanged.
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import importlib.metadata
import shutil
from pathlib import Path

VERSION="0.32.0"
PATH="models/deepseek_v41/common/engram.py"
ORIGINAL_SHA="236235dfa82fd74a4c3a77a69bdfc08d1b20f021"


def oid(data):
    return hashlib.sha1(f"blob {len(data)}".encode()+bytes([0])+data).hexdigest()


def change(src,old,new):
    count=src.count(old)
    if count!=1:
        raise RuntimeError(f"Expected unique Engram anchor, found {count}: {old[:140]!r}")
    return src.replace(old,new)


def patch(root:Path):
    dest=root/PATH
    data=dest.read_bytes()
    if oid(data)!=ORIGINAL_SHA:
        raise RuntimeError(f"Unpinned Engram source {oid(data)} != {ORIGINAL_SHA}")
    s=data.decode("utf8")
    s=change(s,"from vllm.triton_utils import tl, triton\n",
        "from vllm.triton_utils import tl, triton\n"+
        "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32\n")
    s=change(s,
        "    PACKED: tl.constexpr,\n    OUT_STRIDE: tl.constexpr,\n):\n",
        "    PACKED: tl.constexpr,\n    OUT_STRIDE: tl.constexpr,\n    SOFTWARE_FP8: tl.constexpr,\n):\n")
    s=change(s,
        """        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )""",
        """        if SOFTWARE_FP8:
            # An E4M3 data buffer in pinned DRAM is a byte buffer to SM80.
            # Preserve the raw bits and decode to f32 before applying E8M0.
            weight_u8 = weight.to(tl.pointer_type(tl.uint8))
            values = tl.load(
                weight_u8 + local[:, None] * DIM + cols[None, :],
                mask=owned[:, None],
                other=0,
            )
        else:
            values = tl.load(
                weight + local[:, None] * DIM + cols[None, :],
                mask=owned[:, None],
                other=0.0,
            )""")
    s=change(s,
        """            tl.store(
                out + dst[:, None] + cols[None, :],
                values.to(tl.uint8, bitcast=True),
                mask=valid[:, None],
            )""",
        """            if SOFTWARE_FP8:
                packed_values = values
            else:
                packed_values = values.to(tl.uint8, bitcast=True)
            tl.store(
                out + dst[:, None] + cols[None, :],
                packed_values,
                mask=valid[:, None],
            )""")
    s=change(s,
        """            tl.store(
                out + dst[:, None] * DIM + cols[None, :],
                (values.to(tl.float32) * scale).to(tl.bfloat16),
                mask=valid[:, None],
            )""",
        """            if SOFTWARE_FP8:
                value_f32 = _decode_fp8_f32(values, False)
            else:
                value_f32 = values.to(tl.float32)
            tl.store(
                out + dst[:, None] * DIM + cols[None, :],
                (value_f32 * scale).to(tl.bfloat16),
                mask=valid[:, None],
            )""")
    s=change(s,
        """            PACKED=out.dtype == torch.uint8,
            OUT_STRIDE=out.stride(0),
        )""",
        """            PACKED=out.dtype == torch.uint8,
            OUT_STRIDE=out.stride(0),
            SOFTWARE_FP8=(
                current_platform.is_cuda()
                and current_platform.is_device_capability_family(80)
            ),
        )""")
    compile(s,str(dest),"exec")
    return dest,s


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path)
    p.add_argument("--skip-version-check",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    args=p.parse_args()
    if not args.skip_version_check and importlib.metadata.version("vllm")!=VERSION:
        raise RuntimeError("vLLM version mismatch")
    if args.source_root:
        root=args.source_root.resolve()
    else:
        import vllm
        root=Path(vllm.__file__).resolve().parent
    dest,new=patch(root)
    if args.dry_run:
        print("STAGE4_SM80_ENGRAM_SOURCE_PLAN_PASS")
        return
    backup=dest.with_name(dest.name+".sm80-v0320.original")
    if backup.exists():
        raise RuntimeError("Refusing repeated Engram stage4 patch")
    shutil.copy2(dest,backup)
    try:
        dest.write_text(new)
        if not compileall.compile_file(str(dest),quiet=1,force=True):
            raise RuntimeError("Engram stage4 did not compile")
    except Exception:
        shutil.move(str(backup),str(dest))
        raise
    print("STAGE4_SM80_ENGRAM_SOURCE_PASS_NOT_GPU_VALIDATED")


if __name__=="__main__":
    main()
