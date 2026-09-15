#!/usr/bin/env python3
from __future__ import annotations

import compileall
import hashlib
import shutil
import subprocess
from pathlib import Path

import vllm

PATCH_DIR = Path("/opt/sm80-patch")
PATCH = PATCH_DIR / "sm80-cu130-minimal.patch"
SHA_FILE = PATCH_DIR / "sm80-cu130-minimal.patch.sha256"
MANIFEST = PATCH_DIR / "sm80-cu130-minimal.manifest.txt"
ROOT = Path(vllm.__file__).resolve().parent

for p in (PATCH, SHA_FILE, MANIFEST):
    if not p.is_file():
        raise SystemExit(f"ERROR: missing verified SM80 patch input: {p}")

expected_sha = SHA_FILE.read_text(encoding="utf-8").split()[0]
actual_sha = hashlib.sha256(PATCH.read_bytes()).hexdigest()
if actual_sha != expected_sha:
    raise SystemExit(
        f"ERROR: minimal patch SHA256 mismatch: {actual_sha} != {expected_sha}"
    )

manifest_lines = [
    line.strip() for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line.strip()
]
if not manifest_lines or manifest_lines[0] != "changed_files=25":
    raise SystemExit("ERROR: verified minimal manifest must contain exactly 25 changed files")
paths = [line for line in manifest_lines if "/" in line and not line.startswith("patch_bytes=")]
if len(paths) != 25 or len(set(paths)) != 25:
    raise SystemExit(f"ERROR: malformed minimal patch manifest: {len(paths)} unique paths")

forbidden = (
    "models/deepseek_v4/amd/",
    "models/deepseek_v4/cpu/",
    "models/deepseek_v4/xpu/",
    "models/deepseek_v4_1/amd/",
)
for rel in paths:
    if rel.startswith(forbidden):
        raise SystemExit(f"ERROR: non-A100 backend leaked into production patch: {rel}")

required = {
    "models/deepseek_v4_1/ampere/ampere_sparse.py",
    "models/deepseek_v4_1/nvidia/model.py",
    "models/deepseek_v4_1/common/engram.py",
    "v1/attention/ops/fp8_sm80.py",
    "v1/attention/ops/mqa_logits_triton.py",
    "v1/attention/backends/mla/indexer.py",
}
missing_required = sorted(required - set(paths))
if missing_required:
    raise SystemExit(f"ERROR: minimal SM80 patch is missing required paths: {missing_required}")

patch_paths: set[str] = set()
for line in PATCH.read_text(encoding="utf-8").splitlines():
    if not line.startswith("+++ b/"):
        continue
    rel = line[len("+++ b/") :]
    if rel == "/dev/null" or rel.startswith("/") or ".." in Path(rel).parts:
        raise SystemExit(f"ERROR: unsafe path in minimal patch: {rel}")
    patch_paths.add(rel)
    (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)

if patch_paths != set(paths):
    raise SystemExit(
        "ERROR: patch paths do not match verified manifest: "
        f"patch_only={sorted(patch_paths - set(paths))} "
        f"manifest_only={sorted(set(paths) - patch_paths)}"
    )

patch_bin = shutil.which("patch")
if patch_bin is None:
    raise SystemExit("ERROR: official image does not provide the `patch` command")

with PATCH.open("rb") as src:
    subprocess.run(
        [patch_bin, "--batch", "--forward", "-p1", "-d", str(ROOT)],
        stdin=src,
        check=True,
    )

if not compileall.compile_dir(str(ROOT), quiet=1, force=False):
    raise SystemExit("ERROR: Python compile check failed after minimal SM80 patch")

for rel in required:
    if not (ROOT / rel).is_file():
        raise SystemExit(f"ERROR: required patched runtime file missing after apply: {rel}")

(PATCH_DIR / "APPLIED").write_text(
    f"target={ROOT}\npatch_sha256={actual_sha}\nchanged_files=25\n",
    encoding="utf-8",
)
print(f"Applied verified 25-file SM80 patch to {ROOT}")
print(f"patch_sha256={actual_sha}")
