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

required_overlay = [
    DST / "v1/attention/ops/fp8_sm80.py",
    DST / "models/deepseek_v4_1/ampere/ampere_sparse.py",
    DST / "v1/attention/backends/mla/indexer.py",
]
missing_overlay = [str(p) for p in required_overlay if not p.exists()]
if missing_overlay:
    raise SystemExit(f"SM80 overlay incomplete after copy: {missing_overlay}")

if not compileall.compile_dir(str(DST), quiet=1, force=False):
    raise SystemExit("Python compile check failed after SM80 overlay")

marker = Path("/opt/sm80-overlay/APPLIED")
marker.write_text(f"target={DST}\nfiles={copied}\n", encoding="utf-8")
print(f"Applied {copied} SM80 overlay files to {DST}")
