#!/usr/bin/env python3
"""Generate a compact unified patch between two vLLM source trees.

Only UTF-8 source/config text is emitted. Python bytecode and cache files are
ignored. The output uses a/<relative-path> and b/<relative-path> headers so it
can be applied from the installed vllm package root with:

    patch -p1 < sm80-cu129.patch
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

IGNORE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
IGNORE_SUFFIXES = {".pyc", ".pyo", ".so", ".a", ".o", ".cubin", ".fatbin"}
TEXT_SUFFIXES = {".py", ".pyi", ".json", ".toml", ".yaml", ".yml", ".txt", ".md"}


def files(root: Path) -> dict[Path, Path]:
    result: dict[Path, Path] = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        if p.suffix in IGNORE_SUFFIXES:
            continue
        # The SM80 overlay is Python source. Restricting to known text formats
        # prevents accidental binary diffs from the installed package.
        if p.suffix not in TEXT_SUFFIXES:
            continue
        result[rel] = p
    return result


def read_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("patched", type=Path)
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    baseline = args.baseline.resolve()
    patched = args.patched.resolve()
    if not baseline.is_dir() or not patched.is_dir():
        raise SystemExit("baseline and patched must both be directories")

    a = files(baseline)
    b = files(patched)
    changed: list[Path] = []
    chunks: list[str] = []

    for rel in sorted(set(a) | set(b)):
        old = read_lines(a.get(rel))
        new = read_lines(b.get(rel))
        if old == new:
            continue
        changed.append(rel)
        from_name = f"a/{rel.as_posix()}" if rel in a else "/dev/null"
        to_name = f"b/{rel.as_posix()}" if rel in b else "/dev/null"
        diff = difflib.unified_diff(
            old,
            new,
            fromfile=from_name,
            tofile=to_name,
            n=3,
            lineterm="\n",
        )
        chunks.extend(diff)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(chunks), encoding="utf-8")

    print(f"changed_files={len(changed)}")
    print(f"patch_bytes={args.output.stat().st_size}")
    for rel in changed:
        print(rel.as_posix())

    if not changed:
        raise SystemExit("refusing to emit an empty patch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
