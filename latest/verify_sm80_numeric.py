#!/usr/bin/env python3
"""Independent SM80 numerical oracle for V4.1 sparse attention.

Must run on a *real* A100 with the v0.32.0 + SM80 stage1+stage2 patch.
No model weights needed. This test is REQUIRED but NOT proof of end-to-end
model accuracy: the DeepGEMM-free indexer is not implemented in stage2.
"""
from __future__ import annotations

import math
import sys

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.fp8_sm80 import (
    _decode_fp8_f32,
    _encode_e4m3fn_u8,
)

D = 512
NOPE = 448
ROPE = 64
HEADS = 8
SMALL = 0.125


@triton.jit
def _sm80_encode_reference_kernel(inp, out, NUM: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(inp + offs, mask=offs < NUM, other=0.0)
    y = _encode_e4m3fn_u8(x)
    tl.store(out + offs, y, mask=offs < NUM)


@triton.jit
def _sm80_decode_reference_kernel(inp, out, NUM: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(inp + offs, mask=offs < NUM, other=0)
    y = _decode_fp8_f32(x, False)
    tl.store(out + offs, y, mask=offs < NUM)


def expect_close(a: torch.Tensor, b: torch.Tensor, label: str,
                 atol: float = 0.016, rtol: float = 0.02) -> None:
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        raise AssertionError(f"{label}: NaN or Inf output")
    try:
        torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)
    except AssertionError as exc:
        max_abs = (a.float() - b.float()).abs().max().item()
        diff_rms = ((a.float() - b.float()) ** 2).mean().sqrt().item()
        raise AssertionError(
            f"{label}: max_abs={max_abs:.7g}, rms={diff_rms:.7g}\n{exc}"
        ) from exc


def verify_fp8_conversion(device: torch.device) -> None:
    torch.manual_seed(20261018)
    boundaries = torch.tensor(
        [-600., -448., -448.01, -240., -1., -0.5, -0.125,
         -0.015625, -0.0078125, -0.00390625, -0.001953125,
         -0.0009765625, -0.0, 0., 0.0009765625, 0.001953125,
         0.00390625, 0.0078125, 0.015625, 0.125, 0.5, 1.,
         240., 448., 448.01, 600.], dtype=torch.float32,
    )
    # Include values on both sides of normal/subnormal boundaries, ties,
    # long-tail values, and exact zeros. Avoid nonfinite reference ambiguity.
    ties = torch.tensor([
        math.ldexp(1.0 + (k + 0.5) / 8.0, exp)
        for exp in range(-7, 9) for k in range(7)
    ], dtype=torch.float32)
    vals = torch.cat([
        boundaries,
        torch.linspace(-500, 500, 8192),
        torch.randn(15000, dtype=torch.float32) * 64.,
        ties, -ties,
    ], dim=0).contiguous()
    test_vals = vals.to(device)
    expected = vals.to(torch.float8_e4m3fn).view(torch.uint8).to(device)
    actual = torch.empty_like(expected)
    _sm80_encode_reference_kernel[(triton.cdiv(vals.numel(), 256),)](
        test_vals, actual, vals.numel(), BLOCK=256,
    )
    torch.cuda.synchronize()
    mismatch = (expected != actual).nonzero().flatten()
    if mismatch.numel():
        index = mismatch[:12]
        raise AssertionError(
            "E4M3FN RNE encode is not bitwise identical to torch reference: "
            f"count={mismatch.numel()}, vals={vals[index.cpu()].tolist()}, "
            f"expected={expected[index].tolist()}, actual={actual[index].tolist()}"
        )
    print(f"FP8_RNE_ENCODE_BITWISE_PASS n={vals.numel()}")
    codes = torch.arange(256, device=device, dtype=torch.uint8)
    result = torch.empty((256,), device=device, dtype=torch.float32)
    _sm80_decode_reference_kernel[(1,)](codes, result, 256, BLOCK=256)
    reference = torch.arange(256, dtype=torch.uint8).view(
        torch.float8_e4m3fn
    ).to(torch.float32).to(device)
    finite = torch.isfinite(reference)
    if not torch.equal(result[finite], reference[finite]):
        raise AssertionError("E4M3FN decode not bitwise exact for all finite bytes")
    if not torch.isnan(result[~finite]).all():
        raise AssertionError("E4M3FN NaN codewords not retained")
    print("FP8_BYTE_DECODE_EXACT_PASS n=256")


def quantized_page_cache(rows: torch.Tensor, block_size: int) -> tuple[
    torch.Tensor, torch.Tensor
]:
    """Pack the v4/SM80 584-byte cache independently from the vLLM writer.

    Page = all token data (576 bytes per row), followed by all scales
    (8 bytes per row). This catches byte offsets and UE8M0 scale mistakes.
    """
    assert rows.shape[1] == D and rows.dtype == torch.bfloat16
    n = rows.shape[0]
    cache = torch.zeros(
        (triton.cdiv(n, block_size), block_size, 584),
        device=rows.device, dtype=torch.uint8,
    )
    buf = cache.flatten()
    decoded = torch.empty_like(rows, dtype=torch.float32)
    for i in range(n):
        block, off = divmod(i, block_size)
        page = block * block_size * 584
        kv = rows[i]
        tiles = kv[:NOPE].float().reshape(7, 64)
        exp = torch.ceil(
            torch.log2(tiles.abs().amax(dim=1).clamp(min=1e-4) / 448.0)
        )
        scale = torch.exp2(exp)
        code = (tiles / scale[:, None]).clamp(
            -448.0, 448.0
        ).to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
        buf[page + off * 576:page + off * 576 + NOPE] = code
        buf[page + off * 576 + NOPE:page + off * 576 + 576] = (
            kv[NOPE:].contiguous().view(torch.uint8)
        )
        buf[page + block_size * 576 + off * 8:
            page + block_size * 576 + off * 8 + 7] = (
                exp + 127.
            ).clamp(0, 255).to(torch.uint8)
        decoded[i, :NOPE] = code.view(torch.float8_e4m3fn).float().reshape(
            7, 64
        ).mul(scale[:, None]).reshape(-1)
        decoded[i, NOPE:] = kv[NOPE:].float()
    return cache, decoded


def sparse_reference(q: torch.Tensor, main: torch.Tensor,
                     main_rows: list[list[int]], extra: torch.Tensor | None,
                     extra_rows: list[list[int]] | None,
                     sink: torch.Tensor | None, scale: float) -> torch.Tensor:
    out = torch.zeros(q.shape, dtype=torch.float32, device=q.device)
    for b in range(q.shape[0]):
        items = [main[main_rows[b]]] if main_rows[b] else []
        if extra is not None and extra_rows is not None and extra_rows[b]:
            items.append(extra[extra_rows[b]])
        if not items:
            continue
        kv = torch.cat(items, dim=0).float()
        scores = (q[b].float() @ kv.transpose(0, 1)) * scale
        if sink is not None:
            scores = torch.cat([scores, sink[:, None].float()], dim=-1)
            probs = torch.softmax(scores, dim=-1)[:, :-1]
        else:
            probs = torch.softmax(scores, dim=-1)
        out[b] = probs @ kv
    return out.to(q.dtype)


def inverse_rope_reference(
    x: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor,
) -> torch.Tensor:
    out = x.float().clone()
    cos, sin = cos_sin[positions].float().chunk(2, dim=-1)
    left = out[..., NOPE::2].clone()
    right = out[..., NOPE + 1::2].clone()
    out[..., NOPE::2] = left * cos[:, None] + right * sin[:, None]
    out[..., NOPE + 1::2] = right * cos[:, None] - left * sin[:, None]
    return out.to(x.dtype)


def verify_sparse_mla(device: torch.device) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_inverse_rope_rows_,
        rocm_sparse_attn_decode,
        rocm_sparse_attn_prefill,
    )

    torch.manual_seed(1390)
    # Intentionally fragmented logical-to-physical slots, unequal topk lengths,
    # zero-pad sentinels, repeated keys, multi-page cache and sink values.
    q = (torch.randn(2, HEADS, D, device=device) * SMALL).to(torch.bfloat16)
    src_main = (torch.randn(13, D, device=device) * SMALL).to(torch.bfloat16)
    src_extra = (torch.randn(7, D, device=device) * SMALL).to(torch.bfloat16)
    main_cache, main_decoded = quantized_page_cache(src_main, block_size=8)
    extra_cache, extra_decoded = quantized_page_cache(src_extra, block_size=8)
    main_rows = [[0, 9, 4, 12, 9], [5, 11, 2]]
    extra_rows = [[6, 1], [3, 0, 4]]
    width_main, width_extra = 6, 4
    main_idx = torch.full((2, width_main), -1, dtype=torch.int32, device=device)
    extra_idx = torch.full((2, width_extra), -1, dtype=torch.int32, device=device)
    for b in range(2):
        main_idx[b, :len(main_rows[b])] = torch.tensor(
            main_rows[b], dtype=torch.int32, device=device
        )
        extra_idx[b, :len(extra_rows[b])] = torch.tensor(
            extra_rows[b], dtype=torch.int32, device=device
        )
    main_lens = torch.tensor([len(r) for r in main_rows],
                             dtype=torch.int32, device=device)
    extra_lens = torch.tensor([len(r) for r in extra_rows],
                              dtype=torch.int32, device=device)
    sink = torch.linspace(-0.30, 0.30, HEADS, device=device, dtype=torch.float32)
    expected = sparse_reference(q, main_decoded, main_rows, extra_decoded,
                                extra_rows, sink, D**-0.5)
    out = torch.empty_like(q)
    pos = torch.tensor([4, 11], dtype=torch.int64, device=device)
    angles = torch.randn(16, ROPE // 2, device=device)
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1).contiguous()
    rotated_rows = rocm_sparse_attn_decode(
        q=q,
        kv_cache=extra_cache,
        swa_k_cache=main_cache,
        swa_only=False,
        topk_indices=extra_idx,
        topk_lens=extra_lens,
        swa_indices=main_idx,
        swa_lens=main_lens,
        swa_ragged_indices=None,
        swa_ragged_indptr=None,
        topk_ragged_indices=None,
        topk_ragged_indptr=None,
        attn_sink=sink,
        scale=D ** -0.5,
        head_dim=D,
        nope_head_dim=NOPE,
        rope_head_dim=ROPE,
        output=out,
        inv_rope_positions=pos,
        inv_rope_cos_sin_cache=cos_sin,
    )
    if rotated_rows != 0:
        raise AssertionError(
            f"SM80 sparse fallback falsely claimed inverse RoPE on {rotated_rows} rows"
        )
    torch.cuda.synchronize()
    expect_close(out, expected, "sparse decode incl. sink/topk", .02, .02)
    raw = out.clone()
    rocm_inverse_rope_rows_(out, pos, cos_sin, ROPE)
    ref_rotated = inverse_rope_reference(expected, pos, cos_sin)
    expect_close(out, ref_rotated, "inverse RoPE in real consumer", .02, .02)
    if torch.allclose(raw[..., NOPE:], out[..., NOPE:], rtol=1e-4, atol=1e-4):
        raise AssertionError("inverse RoPE test is degenerate (no rotation)")
    if not torch.equal(raw[..., :NOPE], out[..., :NOPE]):
        raise AssertionError("inverse RoPE modified the NoPE dimensions")
    print("SPARSE_DECODE_AND_INVERSE_ROPE_REFERENCE_PASS")

    # No extra cache: sliding-window-only (SWA) branch, including tail/mask.
    expected_swa = sparse_reference(q, main_decoded, main_rows, None, None,
                                    sink, D**-0.5)
    swa = torch.empty_like(q)
    nrot = rocm_sparse_attn_decode(
        q=q, kv_cache=None, swa_k_cache=main_cache, swa_only=True,
        topk_indices=None, topk_lens=None, swa_indices=main_idx,
        swa_lens=main_lens, swa_ragged_indices=None,
        swa_ragged_indptr=None, topk_ragged_indices=None,
        topk_ragged_indptr=None, attn_sink=sink, scale=D**-0.5,
        head_dim=D, nope_head_dim=NOPE, rope_head_dim=ROPE,
        output=swa,
    )
    if nrot != 0:
        raise AssertionError("SWA-only falsely reports inverse RoPE")
    expect_close(swa, expected_swa, "SWA-only", .02, .02)
    print("SPARSE_SWA_ONLY_REFERENCE_PASS")

    # Prefill independently references BF16 Q/K, with explicit selected keys.
    prefill_keys = torch.cat([src_main, src_extra], dim=0)
    prefill_index = torch.tensor([
        [2, 14, 3, -1, -1],
        [19, 7, 11, 14, -1],
    ], device=device, dtype=torch.int32)
    prefill_lens = torch.tensor([3, 4], device=device, dtype=torch.int32)
    prefill_output = torch.empty_like(q)
    rocm_sparse_attn_prefill(
        q=q,
        kv=prefill_keys[:, None, :],
        indices=prefill_index,
        topk_length=prefill_lens,
        scale=D**-0.5, head_dim=D, nope_head_dim=NOPE,
        rope_head_dim=ROPE, attn_sink=sink, output=prefill_output,
    )
    prefill_expected = sparse_reference(
        q, prefill_keys.float(), [[2, 14, 3], [19, 7, 11, 14]],
        None, None, sink, D**-0.5,
    )
    expect_close(prefill_output, prefill_expected, "sparse prefill", .02, .02)
    print("SPARSE_PREFILL_REFERENCE_PASS")


def main() -> None:
    import importlib.metadata as md
    if md.version("vllm") != "0.32.0":
        raise RuntimeError("expected pinned upstream vLLM 0.32.0")
    if not torch.cuda.is_available():
        raise RuntimeError("Requires CUDA A100; refusing to skip numeric tests")
    device = torch.device("cuda", 0)
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    if cap != (8, 0) or not any(x in name for x in ("A100", "A800")):
        raise RuntimeError(f"Only NVIDIA A100/A800 SM80 supported, got {name} {cap}")
    verify_fp8_conversion(device)
    verify_sparse_mla(device)
    print("SM80_NUMERIC_PARITY_PASS_NOT_FULL_MODEL_VALIDATION")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SM80_NUMERIC_PARITY_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
