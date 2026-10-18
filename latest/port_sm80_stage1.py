#!/usr/bin/env python3
"""Stage-1 source port of vLLM v0.32.0 DeepSeek V4.1 SM80 backend.

Purposefully limited to backend registration and vendored software fallbacks.
Additional cache/indexer/Engram SM80 changes are REQUIRED before inference
should be considered functional; see README's bring-up gates.
Never run against an unknown vLLM version or silently patch it.
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import importlib.metadata
import shutil
from pathlib import Path

EXPECTED_VERSION = "0.32.0"
PINNED_BLOBS = {
    "models/deepseek_v41/nvidia/model.py": "8ec0155f736a118e7a20370c3cd1d2a7a06cff87",
    "v1/attention/backends/registry.py": "207c112a384c1e1450362c7c91ac31893fd783bd",
    "model_executor/kernels/linear/scaled_mm/cutlass.py": "a541debfe9f3595f31d08e7685725d18b43c64ac",
}
ROOT_OF_REPO = Path(__file__).resolve().parent.parent
FILES_TO_COPY = {
    "latest/ampere_sparse.py": "models/deepseek_v41/ampere/ampere_sparse.py",
    "patches/vendor/vllm/v1/attention/ops/fp8_sm80.py": "v1/attention/ops/fp8_sm80.py",
    "patches/vendor/vllm/v1/attention/ops/mqa_logits_triton.py": "v1/attention/ops/mqa_logits_triton.py",
}


def git_blob_sha(content: bytes) -> str:
    """SHA1 matching the git blob OID published at vLLM tag v0.32.0."""
    hdr = f"blob {len(content)}".encode() + bytes([0])
    return hashlib.sha1(hdr + content).hexdigest()


def replace_once(source: str, before: str, after: str, rel: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{rel}: expected one unpatched anchor, found {count}")
    return source.replace(before, after, 1)


def verify_vllm_version() -> None:
    version = importlib.metadata.version("vllm")
    if version != EXPECTED_VERSION:
        raise RuntimeError(f"Refusing to patch vLLM {version}, expected {EXPECTED_VERSION}")


def build_changes(root: Path) -> dict[Path, bytes]:
    missing = [f for f in FILES_TO_COPY if not (ROOT_OF_REPO / f).is_file()]
    if missing:
        raise RuntimeError(f"Missing pinned reference files: {missing}")
    updates: dict[Path, bytes] = {}
    for rel, expected_oid in PINNED_BLOBS.items():
        path = root / rel
        raw = path.read_bytes()
        actual = git_blob_sha(raw)
        if actual != expected_oid:
            raise RuntimeError(
                f"Refusing stale patch: {rel} git-blob {actual} != "
                f"upstream v0.32.0 pinned {expected_oid}"
            )
        src = raw.decode("utf-8")
        if rel.endswith("nvidia/model.py"):
            anchor = "    device_capability = current_platform.get_device_capability()\n"
            src = replace_once(src, anchor, anchor +
                "    if device_capability is not None and device_capability.major == 8:\n"
                "        if backend not in (None, AttentionBackendEnum.TRITON_MLA_SPARSE_DSV41):\n"
                "            raise ValueError(\"A100 SM80 requires TRITON_MLA_SPARSE_DSV41\")\n"
                "        from vllm.models.deepseek_v41.ampere.ampere_sparse import (\n"
                "            DeepseekV41AmpereMLAAttention,\n"
                "        )\n"
                "        return DeepseekV41AmpereMLAAttention\n", rel)
        elif rel.endswith("registry.py"):
            anchor = "    FLASHINFER_MLA_SPARSE_DSV41 = (\n"
            src = replace_once(src, anchor,
                "    TRITON_MLA_SPARSE_DSV41 = (\n"
                "        \"vllm.models.deepseek_v41.ampere.ampere_sparse.\"\n"
                "        \"DeepseekV41AmpereMLASparseBackend\"\n"
                "    )\n" + anchor, rel)
        elif rel.endswith("scaled_mm/cutlass.py"):
            anchor = (
                "        if not cutlass_fp8_supported():\\n"
                "            return False, \\"CUTLASS FP8 kernels not available\\"\\n"
            )
            replacement = (
                "        if compute_capability is None:\\n"
                "            capability = current_platform.get_device_capability()\\n"
                "            compute_capability = -1 if capability is None else capability.to_int()\\n"
                "        if compute_capability < 89:\\n"
                "            return False, \\"CUTLASS FP8 tensor-core MMA needs SM89+\\"\\n"
            ) + anchor
            # Anchor occurs only in the FP8 class, not in Int8.
            src = replace_once(src, anchor, replacement, rel)
        updates[path] = src.encode("utf-8")
    for from_rel, dest_rel in FILES_TO_COPY.items():
        dest = root / dest_rel
        if dest.exists():
            raise RuntimeError(
                f"Refusing to replace an upstream file: {dest_rel}"
            )
        updates[dest] = (ROOT_OF_REPO / from_rel).read_bytes()
    init = root / "models/deepseek_v41/ampere/__init__.py"
    if init.exists():
        raise RuntimeError(f"Refusing to replace upstream Ampere package: {init}")
    updates[init] = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "\"\"\"DeepSeek V4.1 CUDA SM80 backend.\"\"\"\n"
    ).encode()
    return updates


def apply(root: Path, dry_run: bool) -> None:
    updates = build_changes(root)
    print(f"pin=vLLM-{EXPECTED_VERSION} updates={len(updates)}")
    for path, content in updates.items():
        print(f"  {path.relative_to(root)} bytes={len(content)}")
    if dry_run:
        print("STAGE1_PATCH_PLAN_PASS (no files modified)")
        return
    backups: list[Path] = []
    created: list[Path] = []
    try:
        for path, content in updates.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                backup = path.with_name(path.name + ".sm80-v0320.orig")
                if backup.exists():
                    raise RuntimeError(f"Refusing overwrite of backup {backup}")
                shutil.copy2(path, backup)
                backups.append(backup)
            else:
                created.append(path)
            path.write_bytes(content)
        for path in updates:
            if path.suffix == ".py":
                ok = compileall.compile_file(str(path), quiet=1, force=True)
                if not ok:
                    raise RuntimeError(f"Python compile failed at {path}")
    except Exception:
        for original in backups:
            shutil.move(str(original), str(original).removesuffix(".sm80-v0320.orig"))
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print("STAGE1_SM80_BACKEND_SOURCE_PATCHED (NOT INFERENCE VALIDATED)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-version-check", action="store_true",
                        help="Only for offline tests with an exact v0.32.0 source tree")
    args = parser.parse_args()
    if not args.skip_version_check:
        verify_vllm_version()
    if args.vllm_root:
        root = args.vllm_root.resolve()
    else:
        import vllm
        root = Path(vllm.__file__).resolve().parent
    apply(root, args.dry_run)


if __name__ == "__main__":
    main()
