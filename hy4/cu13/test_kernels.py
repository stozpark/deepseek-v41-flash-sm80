#!/usr/bin/env python3
"""Physical SM80 checks; failures must block full-model testing."""
import torch
from vllm.models.hy_v4.nvidia.sm80_sparse import apply_attention_sink
from vllm.models.hy_v4.nvidia.sm80_indexer_quant import quantize_indexer_q
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton, fp8_paged_mqa_logits_triton,
)


def test_sink():
    torch.manual_seed(17)
    for heads in (8, 16):
        rows, topk, dim = 5, 2048, 576
        q = torch.randn(rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(2200, dim, device="cuda", dtype=q.dtype)
        ids = torch.randperm(kv.shape[0], device="cuda")[:topk].int().repeat(rows, 1)
        ids[0] = -1
        ids[1, 1:] = -1
        ids[2, ::7] = -1
        ids[3, topk // 2:] = -1
        sinks = torch.linspace(-40, 40, heads, device="cuda")
        keys = kv.float().cpu()[ids.long().cpu().clamp_min(0)]
        scores = q.float().cpu().bmm(keys.transpose(1, 2)) * (256**-0.5)
        scores.masked_fill_(ids.cpu()[:, None, :] < 0, -torch.inf)
        scores = torch.cat(
            [scores, sinks.cpu()[None, :, None].expand(rows, heads, 1)], dim=-1
        )
        expected = scores.softmax(-1)[..., :-1].bmm(keys[..., :512])
        ref_lse = scores.logsumexp(-1)
        for splits in (1, 4):
            out, lse = triton_mla_sparse_fwd(
                q, kv, ids, sm_scale=256**-0.5, kv_lora_rank=512,
                return_lse=True, num_kv_splits=splits,
            )
            assert torch.isfinite(out).all() and not out[0].any()
            out, total_lse = apply_attention_sink(out, lse, sinks)
            assert torch.isfinite(out).all() and not out[0].any()
            torch.testing.assert_close(out.float().cpu(), expected, atol=0.015, rtol=0.025)
            torch.testing.assert_close(total_lse.cpu(), ref_lse, atol=3e-4, rtol=3e-4)
            print("sink PASS", heads, splits)


def test_quant():
    torch.manual_seed(29)
    q = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    q[0].zero_()
    q[1] *= 1e-5
    q[2] *= 100
    for ue8m0 in (False, True):
        actual, scales = quantize_indexer_q(q, 128, ue8m0)
        x = q.float().cpu()
        ref_scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) * (1/448.0)
        if ue8m0:
            ref_scale = torch.exp2(torch.ceil(torch.log2(ref_scale)))
        ref = (x / ref_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        torch.testing.assert_close(scales.cpu(), ref_scale, rtol=1e-6, atol=0)
        decoded = actual.view(torch.uint8).cpu().view(torch.float8_e4m3fn).float() * scales.cpu()
        expected = ref.float() * ref_scale
        torch.testing.assert_close(decoded, expected, atol=0.0, rtol=0.14)
        agreement = (actual.view(torch.uint8).cpu() == ref.view(torch.uint8)).float().mean()
        assert agreement >= .999, f"FP8 byte agreement {agreement.item()}"
        print("query FP8 PASS", ue8m0, "byte agreement", agreement.item())


def test_mqa():
    torch.manual_seed(31)
    page, dim, heads = 64, 128, 32
    # Generate FP8 on CPU, transfer bytes. Avoid native FP8 conversion on SM80.
    q_cpu = torch.randn(2, heads, dim).to(torch.float8_e4m3fn)
    k_cpu = torch.randn(4, page, dim).to(torch.float8_e4m3fn)
    q = q_cpu.view(torch.uint8).cuda().view(torch.float8_e4m3fn)
    k = k_cpu.view(torch.uint8).cuda().view(torch.float8_e4m3fn)
    w_cpu = torch.randn(2, heads)
    sc_cpu = (2.0 ** torch.arange(-3, 5)).repeat(4, 8)
    w, sc = w_cpu.cuda(), sc_cpu.cuda()
    starts = torch.tensor([7, 80], device="cuda", dtype=torch.int32)
    ends = torch.tensor([74, 201], device="cuda", dtype=torch.int32)
    pre = fp8_mqa_logits_triton(q, (k.reshape(-1, dim), sc.flatten()), w,
                                starts, ends, 128).cpu()
    for r, (a, b) in enumerate([(7, 74), (80, 201)]):
        keys = k_cpu.float().reshape(-1, dim)[a:b]
        ref = ((q_cpu[r].float() @ keys.T).relu() * w_cpu[r, :, None]).sum(0)
        ref *= sc_cpu.flatten()[a:b]
        torch.testing.assert_close(pre[r, :b-a], ref, rtol=2e-4, atol=5e-3)
    # Non-contiguous page stride and nontrivial physical mapping.
    pages = torch.empty((4, page * (dim + 4) + 32), device="cuda", dtype=torch.uint8)
    cache = pages[:, :-32]
    cache[:, :page * dim] = k.view(torch.uint8).flatten(1)
    cache[:, page * dim:] = sc.view(torch.uint8)
    table_cpu = torch.tensor([[3, 0], [1, 2]], dtype=torch.int32)
    lengths = torch.tensor([[66], [127]], device="cuda", dtype=torch.int32)
    for lens in (lengths, lengths.flatten()):
        dec = fp8_paged_mqa_logits_triton(
            q[:, None], cache.view(4, page, dim + 4), w, lens,
            table_cpu.cuda(), 192,
        ).cpu()
        for r, n in enumerate((66, 127)):
            keys = k_cpu.float()[table_cpu[r].long()].reshape(-1, dim)[:n]
            ref = ((q_cpu[r].float() @ keys.T).relu() * w_cpu[r, :, None]).sum(0)
            ref *= sc_cpu[table_cpu[r].long()].flatten()[:n]
            torch.testing.assert_close(dec[r, :n], ref, rtol=2e-4, atol=5e-3)
    print("prefill/paged MQA valid scores PASS")


def test_native_indexer_insert():
    """Exercise the real CUDA cache insertion, including skipped slots and padded page strides."""
    from vllm import _custom_ops as ops

    torch.manual_seed(53)
    page, dim = 64, 128
    slots = torch.tensor([0, 63, 64, 129, -1, 191], dtype=torch.int64)
    keys = torch.randn(slots.numel(), dim, dtype=torch.bfloat16)
    keys[1].zero_()
    keys[3] *= 1e-6
    backing = torch.full((3, page * (dim + 4) + 32), 0xA5,
                         device="cuda", dtype=torch.uint8)
    cache = backing[:, :-32].view(3, page, dim + 4)
    ops.indexer_k_quant_and_cache(keys.cuda(), cache, slots.cuda(), dim, "ue8m0")
    torch.cuda.synchronize()
    result = backing.cpu()
    expected = torch.full_like(result, 0xA5)
    for i, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, page)
        x = keys[i].float()
        # Native cache op uses 1e-4 minimum amax (query helper uses 1e-10).
        scale = torch.exp2(torch.ceil(torch.log2(x.abs().amax().clamp_min(1e-4) / 448.0)))
        raw = (x / scale).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        expected[block, offset*dim:(offset+1)*dim] = raw
        start = page*dim + offset*4
        expected[block, start:start+4] = scale.reshape(1).view(torch.uint8)
    assert torch.equal(result, expected), "native K insertion bytes/scale/guard mismatch"
    print("native K cache insert PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 0):
        raise RuntimeError("requires physical SM80")
    if not torch.version.cuda or not torch.version.cuda.startswith("13.2"):
        raise RuntimeError("requires pinned CUDA 13.2 PyTorch build")
    test_sink()
    test_quant()
    test_mqa()
    test_native_indexer_insert()
    from test_gpu_integration import test_indexer_chain, test_large_addresses
    test_indexer_chain()
    test_large_addresses()
    torch.cuda.synchronize()
    print("HYV4_SM80_KERNELS=PASS")
