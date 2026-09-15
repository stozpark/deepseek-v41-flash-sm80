#!/usr/bin/env python3
"""Copy only text-source files from the installed vLLM package.

Used by the official-image audit to create small baseline/patched snapshots
without copying CUDA extensions, bytecode, or other large binaries.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import vllm

TEXT_SUFFIXES = {".py", ".pyi", ".json", ".toml", ".yaml", ".yml", ".txt", ".md"}
IGNORE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: snapshot_vllm_text.py OUTPUT_DIR")
    src = Path(vllm.__file__).resolve().parent
    dst = Path(sys.argv[1]).resolve()
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    copied = 0
    total_bytes = 0
    for p in src.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        rel = p.relative_to(src)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)
        copied += 1
        total_bytes += p.stat().st_size

    print(f"vllm_root={src}")
    print(f"snapshot={dst}")
    print(f"files={copied}")
    print(f"bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
