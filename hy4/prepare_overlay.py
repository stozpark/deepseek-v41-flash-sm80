#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare a pinned, clean vLLM source checkout; never patch installed wheels.

Usage: python hy4/prepare_overlay.py /path/to/pinned-vllm
--check validates every source anchor and compiles generated Python syntax
without writing. Actual apply refuses drift and existing overlay files.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    spec = json.loads((HERE / "source_manifest.json").read_text())
    if git(root, "rev-parse", "HEAD") != spec["source_commit"]:
        raise SystemExit("Refusing unpinned vLLM source")
    if git(root, "status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Refusing dirty vLLM checkout")
    pending = {}
    for entry in spec["files"]:
        path = root / entry["path"]
        raw = path.read_bytes()
        if blob_sha(raw) != entry["git_blob_sha"]:
            raise SystemExit(f"Source hash mismatch: {path}")
        content = raw.decode("utf-8")
        for old, new in entry["replacements"]:
            if content.count(old) != 1:
                raise SystemExit(f"Non-unique/missing source anchor: {path}")
            content = content.replace(old, new, 1)
        pending[path] = content
    dest = root / "vllm/models/hy_v4/nvidia"
    for name in ("sm80_sink_math.py", "triton_mla_sparse.py"):
        path = dest / name
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
        pending[path] = (HERE / "overlay" / name).read_text(encoding="utf-8")
    # No writes until all hashes, anchors and syntax have passed.
    for path, content in pending.items():
        ast.parse(content, filename=str(path))
    if not args.check:
        for path, content in pending.items():
            path.write_bytes(content.encode("utf-8"))
    print(f"HY4_OVERLAY_{'CHECK' if args.check else 'APPLIED'}=PASS ({len(pending)} files)")
    print("Source-only; build and physical GPU numerical/serving validation required.")


if __name__ == "__main__":
    main()
