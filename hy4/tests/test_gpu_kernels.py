# SPDX-License-Identifier: Apache-2.0
"""Run after installing the patched, pinned vLLM on a real A100/A800.

pytest -q hy4/tests/test_gpu_kernels.py
No full-checkpoint or serving claims follow from passing these tests.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0),
    reason="requires physical SM80 GPU",
)


@pytest.mark.parametrize("splits", [1, 4])
def test_sparse_sink_parity(splits):
    from vllm.models.hy_v4.nvidia.sm80_sink_math import apply_sink
    from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd

    torch.manual_seed(123)
    q = torch.randn(3, 8, 576, device="cuda", dtype=torch.bfloat16) * 0.25
    kv = torch.randn(96, 576, device="cuda", dtype=torch.bfloat16) * 0.25
    idx = torch.arange(96, device="cuda", dtype=torch.int32).repeat(3, 1)
    idx[0, 7] = -1
    idx[1, 64:] = -1
    idx[2] = -1
    sinks = torch.linspace(-4, 4, 8, device="cuda", dtype=torch.float32)
    scale = 256 ** -0.5  # model's original qk dimension, not absorbed 576
    out, lse = triton_mla_sparse_fwd(
        q, kv, idx, sm_scale=scale, kv_lora_rank=512,
        return_lse=True, num_kv_splits=splits,
    )
    actual = apply_sink(out, lse, sinks)
    keys = kv[idx.clamp(min=0).long()].float()
    score = torch.einsum("thd,tkd->thk", q.float(), keys) * scale
    score = score.masked_fill(idx[:, None, :] < 0, -torch.inf)
    ref_lse = score.logsumexp(-1)
    score = torch.cat((score, sinks.expand(3, 8).unsqueeze(-1)), -1)
    prob = score.softmax(-1)[..., :-1]
    expected = torch.einsum("thk,tkd->thd", prob, keys[..., :512])
    torch.testing.assert_close(lse, ref_lse, atol=0.01, rtol=0.01)
    torch.testing.assert_close(actual.float(), expected, atol=0.008, rtol=0.03)
    assert torch.isfinite(actual).all() and actual[2].count_nonzero() == 0


@pytest.mark.parametrize("is_neox", [True, False])
def test_hyv4_fused_indexer_software_fp8(is_neox):
    from vllm.models.hy_v4.nvidia.indexer_fusion import fused_indexer_q_rope_quant

    torch.manual_seed(17)
    q = torch.randn(5, 32, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
    pos = torch.arange(5, device="cuda", dtype=torch.int64)
    angles = torch.randn(5, 32, device="cuda")
    cache = torch.cat((angles.cos(), angles.sin()), -1).bfloat16()
    q_fp8, w_out = fused_indexer_q_rope_quant(
        pos, q, cache, w, 128 ** -0.5, 32 ** -0.5, is_neox
    )
    # Independent CPU reference of the exact trailing-64 RoPE layout,
    # BF16 rounding, UE8M0 ceil scale and software E4M3 conversion.
    q_cpu, cache_cpu = q.cpu().float(), cache.cpu().float()
    c, s = cache_cpu[:, None, :32], cache_cpu[:, None, 32:]
    if is_neox:
        x0, x1 = q_cpu[..., 64:96], q_cpu[..., 96:128]
    else:
        x0, x1 = q_cpu[..., 64::2], q_cpu[..., 65::2]
    r0 = (x0 * c - x1 * s).bfloat16().float()
    r1 = (x1 * c + x0 * s).bfloat16().float()
    rope = torch.cat((r0, r1), -1) if is_neox else torch.stack((r0, r1), -1).flatten(-2)
    rot = torch.cat((q_cpu[..., :64], rope), -1)
    scale = torch.exp2(torch.ceil(torch.log2(rot.abs().amax(-1).clamp_min(1e-10) / 448.0)))
    expected = (rot / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(q_fp8.cpu().view(torch.uint8), expected.view(torch.uint8))
    expected_w = w.cpu().float() * scale * (128 ** -0.5) * (32 ** -0.5)
    torch.testing.assert_close(w_out.cpu(), expected_w, atol=1e-6, rtol=1e-5)
