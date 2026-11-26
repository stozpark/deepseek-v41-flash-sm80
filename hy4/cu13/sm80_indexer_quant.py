# SPDX-License-Identifier: Apache-2.0
"""Software FP8 query quantization for HYV4 on Ampere."""
import torch
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.common import f32_to_e4m3_uint8


@triton.jit
def _quantize_query(x, out, scales, stride: tl.int64, D: tl.constexpr,
                    UE8M0: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, D)
    v = tl.load(x + row * stride + d).to(tl.float32)
    raw = tl.maximum(tl.max(tl.abs(v)), 1.0e-10) * (1.0 / 448.0)
    scale = tl.exp2(tl.ceil(tl.log2(raw))) if UE8M0 else raw
    byte = f32_to_e4m3_uint8(tl.minimum(tl.maximum(v / scale, -448.0), 448.0))
    tl.store(out + row * D + d, byte)
    tl.store(scales + row, scale)


def quantize_indexer_q(q: torch.Tensor, group_size: int, use_ue8m0: bool):
    if q.ndim != 2 or q.shape[-1] != 128 or group_size != 128:
        raise ValueError("SM80 HYV4 indexer expects one 128-wide group per row")
    if q.dtype not in (torch.bfloat16, torch.float16) or not q.is_cuda:
        raise ValueError("SM80 HYV4 indexer expects CUDA BF16/FP16")
    q = q.contiguous()
    raw = torch.empty(q.shape, device=q.device, dtype=torch.uint8)
    scales = torch.empty((q.shape[0], 1), device=q.device, dtype=torch.float32)
    if q.shape[0]:
        _quantize_query[(q.shape[0],)](
            q, raw, scales, q.stride(0), D=128, UE8M0=use_ue8m0, num_warps=4
        )
    return raw.view(torch.float8_e4m3fn), scales
