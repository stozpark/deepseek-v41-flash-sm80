#!/usr/bin/env python3
"""Verify exact patched wheel sources, native registrations and launch CLI.

The manifest is generated AFTER verify-applied on the pinned source tree.
Keep it with the image/SIF; verify independently on BOTH serving nodes.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess

PIN = "3470b0db1820f699960d1376223602739a0f4ac3"
MODULES = ("attention", "model", "sm80_sparse", "sm80_indexer_quant", "sink_math")
PREFIX = "vllm/models/hy_v4/nvidia/"
PORT_FILES = (
    "image_verify.py", "audited_worker.py", "runtime_preflight.py", "serve.sh",
    "test_kernels.py", "test_gpu_integration.py", "test_marlin.py",
    "reference_checks.py", "make_fixture.py", "serve_fixture.sh",
    "serve_probe.py", "nccl_probe.py", "test_hy4_math.py", "validate_logs.py",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def create_manifest(src, port):
    return {
        "source_commit": PIN, "cuda_toolkit": "13.2.1",
        "modules": {PREFIX + m + ".py": digest(src / (PREFIX + m + ".py"))
                    for m in MODULES},
        "port": {p: digest(port / p) for p in PORT_FILES},
    }


def verify_files(root, port, manifest):
    if manifest.get("source_commit") != PIN or manifest.get("cuda_toolkit") != "13.2.1":
        raise RuntimeError("wrong source/CUDA manifest")
    for key, base, expected in (
        ("modules", root, {PREFIX + m + ".py" for m in MODULES}),
        ("port", port, set(PORT_FILES)),
    ):
        hashes = manifest.get(key, {})
        if set(hashes) != expected:
            raise RuntimeError(f"incomplete {key} manifest")
        for rel, sha in hashes.items():
            if digest(base / rel) != sha:
                raise RuntimeError(f"image/port hash mismatch: {rel}")


def verify_image(manifest_path, gpu=False):
    import torch
    import vllm
    root = Path(vllm.__file__).resolve().parent.parent
    port = Path(__file__).resolve().parent
    manifest = json.loads(manifest_path.read_text())
    verify_files(root, port, manifest)
    if not torch.version.cuda or not torch.version.cuda.startswith("13.2"):
        raise RuntimeError(f"expected cu132 PyTorch, got {torch.version.cuda}")
    if gpu and (not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0)):
        raise RuntimeError("GPU verification requires SM80")
    # Explicit imports even if platform autodetection is unspecified on the
    # build host. These are the stable-ABI modules at the exact source pin.
    for name in ("vllm._C_stable_libtorch", "vllm._moe_C_stable_libtorch"):
        importlib.import_module(name)
    for namespace, op in (
        ("_C", "gptq_marlin_repack"), ("_moe_C", "moe_wna16_marlin_gemm"),
        ("_C_cache_ops", "indexer_k_quant_and_cache"),
        ("_C_cache_ops", "cp_gather_indexer_k_quant_cache"),
    ):
        if not hasattr(getattr(torch.ops, namespace), op):
            raise RuntimeError(f"native op not registered: {namespace}::{op}")
    for name in MODULES:
        mod = importlib.import_module("vllm.models.hy_v4.nvidia." + name)
        expected = (root / (PREFIX + name + ".py")).resolve()
        if Path(mod.__file__).resolve() != expected:
            raise RuntimeError(f"import shadowing: {name}")
    help_text = subprocess.run(["vllm", "serve", "--help=all"], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, timeout=180).stdout
    for flag in ("--worker-cls", "--linear-backend", "--moe-backend",
                 "--nnodes", "--node-rank", "--headless", "--attention-config",
                 "--language-model-only", "--load-format", "--master-addr"):
        if flag not in help_text:
            raise RuntimeError(f"missing pinned CLI flag: {flag}")
    print("HY4_IMAGE_PACKAGE=PASS", torch.__version__, torch.version.cuda,
          "GPU" if gpu else "host import/ABI only", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("--create-from", type=Path)
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args()
    if a.create_from:
        a.manifest.write_text(json.dumps(create_manifest(
            a.create_from.resolve(), Path(__file__).resolve().parent),
            indent=2, sort_keys=True) + "\n")
    else:
        verify_image(a.manifest, a.gpu)
