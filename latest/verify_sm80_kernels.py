#!/usr/bin/env python3
"""Physical A100 SM80 kernel probes; no 510 GB model download required.

Run inside the patched vLLM 0.32.0 CUDA13 SIF with Apptainer --nv. These tests
catch missing backend registration, FP8 conversion and paged-MQA tail bugs.
Passing is **necessary but not sufficient** for inference correctness.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import sys

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("SM80_KERNEL_PROBE_FAIL: " + message)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    require(md.version("vllm") == "0.32.0", "expected exact vLLM 0.32.0")
    cuda = str(torch.version.cuda or "")
    require(cuda and int(cuda.split(".")[0]) >= 13, f"requires CUDA13+, got {cuda}")
    require(torch.cuda.is_available(), "GPU unavailable (missing --nv?)")
    require(0 <= args.gpu < torch.cuda.device_count(), "invalid device index")
    device = torch.device("cuda", args.gpu)
    cap = torch.cuda.get_device_capability(args.gpu)
    require(cap == (8, 0), f"expected SM80, got {cap}")
    print("torch=", torch.__version__, "CUDA=", cuda,
          "device=", torch.cuda.get_device_name(args.gpu))

    # Registration cannot be inferred from Python syntax; force the imports.
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    require(
        "TRITON_MLA_SPARSE_DSV41" in AttentionBackendEnum.__members__,
        "missing v41 SM80 attention backend",
    )
    from vllm.models.deepseek_v41.ampere.ampere_sparse import (
        DeepseekV41AmpereMLAAttention,
        DeepseekV41AmpereMLASparseBackend,
    )
    require(
        DeepseekV41AmpereMLASparseBackend.get_name() == "TRITON_MLA_SPARSE_DSV41",
        "unexpected Ampere sparse backend",
    )
    print("AMPERE_BACKEND_IMPORT_PASS", DeepseekV41AmpereMLAAttention.__name__)

    # No CUTLASS fp8 tensor cores on A100; a BF16/Marlin fallback is mandatory.
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import (
        CutlassFP8ScaledMMLinearKernel,
    )
    ok, msg = CutlassFP8ScaledMMLinearKernel.is_supported(80)
    print("CUTLASS_FP8_SM80=", ok, msg)
    require(not ok, "CUTLASS native FP8 incorrectly enabled on SM80")

    from vllm.model_executor.kernels.linear.mxfp4.marlin import (
        MarlinMxFp4LinearKernel,
    )
    from vllm.model_executor.kernels.linear.mxfp8.marlin import (
        MarlinMxfp8LinearKernel,
    )
    print("MXFP4_MARLIN=", MarlinMxFp4LinearKernel.is_supported(80))
    print("MXFP8_MARLIN=", MarlinMxfp8LinearKernel.is_supported(80))

    # Software E4M3 converter must be loadable on A100; 256 byte LUT.
    from vllm.v1.attention.ops.fp8_sm80 import get_e4m3fn_bf16_lut
    lut = get_e4m3fn_bf16_lut(device)
    torch.cuda.synchronize(args.gpu)
    require(
        lut.shape == (256,) and lut.dtype == torch.bfloat16
        and lut.device == device,
        f"FP8 LUT invalid: shape={lut.shape}, dtype={lut.dtype}, device={lut.device}",
    )
    print("SM80_SOFTWARE_FP8_LUT_PASS")

    # A live Triton paged MQA kernel on a deliberately partial final block.
    from vllm.v1.attention.ops.mqa_logits_triton import (
        fp8_paged_mqa_logits_triton,
    )
    B, NEXT_N, H, D = 1, 1, 16, 128
    block_size = 128
    q = torch.zeros((B, NEXT_N, H, D),
                    dtype=torch.float8_e4m3fn, device=device)
    cache = torch.zeros((1, block_size, 1, D + 4),
                        dtype=torch.uint8, device=device)
    weights = torch.ones((B * NEXT_N, H),
                         dtype=torch.float32, device=device)
    seq_lens = torch.tensor([1], dtype=torch.int32, device=device)
    block_tables = torch.zeros((B, 2), dtype=torch.int32, device=device)
    logits = fp8_paged_mqa_logits_triton(
        q, cache, weights, seq_lens, block_tables, max_model_len=65
    )
    torch.cuda.synchronize(args.gpu)
    require(torch.isfinite(logits[0, 0]).item(), "live MQA logit is not finite")
    require(torch.isneginf(logits[0, 1:]).all().item(),
            "paged MQA invalid tail logits (expected -inf)")
    print("SM80_PAGED_MQA_TAIL_PASS")
    print("STAGE1_SM80_KERNEL_PROBES_PASS_NOT_MODEL_VALIDATION")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__ + ": " + str(exc), file=sys.stderr)
        raise
