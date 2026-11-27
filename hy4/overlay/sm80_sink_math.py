# SPDX-License-Identifier: Apache-2.0
"""Numerical helper: real-key natural-log LSE and learned zero-value sink."""
import torch


def apply_sink(out: torch.Tensor, lse: torch.Tensor,
               sinks: torch.Tensor) -> torch.Tensor:
    """Return [T,H,D] output in its original dtype. Sinks are TP-local [H]."""
    if out.ndim != 3 or lse.shape != out.shape[:2]:
        raise ValueError("expected output [T,H,D] and LSE [T,H]")
    if sinks.ndim != 1 or sinks.shape[0] != out.shape[1]:
        raise ValueError("sink must match the actual local query heads")
    if lse.dtype != torch.float32 or sinks.dtype != torch.float32:
        raise ValueError("LSE and sinks must be FP32")
    if out.device != lse.device or out.device != sinks.device:
        raise ValueError("sink/output/LSE device mismatch")
    # All real keys masked -> lse=-inf; virtual zero-value sink yields zero.
    # The underlying sparse kernel must provide a valid (zero) empty output.
    return (out.float() * torch.sigmoid(lse - sinks)[..., None]).to(out.dtype)
