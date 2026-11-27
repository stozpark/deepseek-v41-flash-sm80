# SPDX-License-Identifier: Apache-2.0
"""Hy4 virtual zero-value sink; operates on natural-log real-key LSE."""
import torch


def apply_attention_sink(out, lse, sinks, *, return_lse=True):
    if out.ndim != 3 or lse.shape != out.shape[:2]:
        raise ValueError("expected [T,H,D] output and [T,H] LSE")
    if sinks.shape != (out.shape[1],) or sinks.dtype != torch.float32:
        raise ValueError("expected one FP32 sink per local head")
    if lse.dtype != torch.float32 or out.device != lse.device or out.device != sinks.device:
        raise ValueError("LSE must be FP32 and all tensors on the same device")
    weight = torch.sigmoid(lse - sinks)
    result = (out.float() * weight.unsqueeze(-1)).to(out.dtype)
    return result, torch.logaddexp(lse, sinks) if return_lse else None
