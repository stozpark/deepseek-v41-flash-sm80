#!/usr/bin/env python3
"""Physical A100 Engram FP8 lookup oracle, no model weights.

The model wrapper MUST pass a byte-typed pointer to Triton on SM80.
Check no-copy FP8->uint8 view; compare live model lookup and raw kernel
against independent CPU FP8/E8M0/BF16 reference. Also validate pinned CPU
UVA table access using vLLM's real accelerator-view helper.
"""
import importlib.metadata
from types import SimpleNamespace

import torch


def main():
    if importlib.metadata.version("vllm")!="0.32.0":
        raise RuntimeError("Requires pinned vLLM v0.32.0")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)!=(8,0):
        raise RuntimeError("Physical A100/A800 SM80 required")

    from vllm.models.deepseek_v41.common.engram import (
        ParallelEngramEmbedding,
        _engram_lookup_kernel,
        _engram_sm80_weight_bytes,
    )
    torch.manual_seed(62197)
    dev=torch.device("cuda",0)
    T,H,N,D,BS=4,2,32,256,32
    # Produce FP8 on CPU, then transfer raw bytecodes; no CUDA FP8 MMA needed.
    cpu_fp8=(torch.randn(N,D,dtype=torch.float32)*0.3).to(
        torch.float8_e4m3fn).contiguous()
    cpu_codes=cpu_fp8.view(torch.uint8)
    table_fp8=cpu_codes.to(dev).view(torch.float8_e4m3fn)
    # This exact dtype transition is essential: passing ptr<fp8e4nv> as
    # Triton kernel arg fails during JIT argument typing on SM80.
    table=_engram_sm80_weight_bytes(table_fp8)
    if (table.dtype != torch.uint8 or table.shape != table_fp8.shape
            or table.data_ptr() != table_fp8.data_ptr()
            or not torch.equal(table, cpu_codes.to(dev))):
        raise AssertionError("E4M3FN->uint8 GPU view is not bit/ptr identical")
    print("ENGRAM_SM80_UINT8_KERNEL_ARGUMENT_PASS")
    sf=torch.randint(124,131,(N,D//BS),dtype=torch.uint8).to(dev)
    ids=torch.tensor([[3,19],[0,6],[31,-1],[35,9]],dtype=torch.int64,device=dev)
    bf16=torch.empty((T,H,D),dtype=torch.bfloat16,device=dev)
    packed=torch.empty((T,H,D+D//BS),dtype=torch.uint8,device=dev)
    # Exercise the actual v0.32.0 model lookup method, not just a manually
    # corrected test call. A real EngramEmbedding constructs the same
    # arguments from _storage() and must never feed the FP8 pointer to JIT.
    fake=SimpleNamespace(
        part_n_hash_cols=H,
        cpu_offload=False,
        _packed=None,
        head_start=0,
        n_hash_cols=H,
        vocab_start_idx=0,
        vocab_end_idx=N,
        dim=D,
        block_size=BS,
        _num_sms=torch.cuda.get_device_properties(dev).multi_processor_count,
        _storage=lambda: (table_fp8, sf),
    )
    for result in (bf16, packed):
        ParallelEngramEmbedding.lookup(fake, ids, result)
    # Also exercise the raw JIT ABI explicitly with a uint8 argument.
    direct=torch.empty_like(bf16)
    _engram_lookup_kernel[(1,)](
        table, sf, ids, ids, direct,
        0, N, T*H, ids.stride(0), ids.stride(1),
        HEAD_START=0, LOCAL_HEADS=H, TOTAL_HEADS=H,
        DIM=D, QUANT_BLOCK=BS, BLOCK_R=16, GRID=1,
        SORTED=False, PACKED=False, OUT_STRIDE=direct.stride(0),
        SOFTWARE_FP8=True,
    )
    torch.cuda.synchronize()
    if not torch.equal(direct,bf16):
        raise AssertionError("model lookup wrapper and raw uint8 ABI disagree")
    decoded=cpu_fp8.float().to(dev)
    for t in range(T):
        for h in range(H):
            idx=int(ids[t,h])
            if idx<0 or idx>=N:
                assert torch.count_nonzero(bf16[t,h])==0, "invalid BF16 lookup not zero"
                assert torch.count_nonzero(packed[t,h])==0, "invalid packed lookup not zero"
                continue
            if not torch.equal(packed[t,h,:D],table[idx].view(torch.uint8)):
                raise AssertionError(f"Engram FP8 bytes mismatch at ({t},{h})")
            if not torch.equal(packed[t,h,D:],sf[idx]):
                raise AssertionError(f"Engram E8M0 scales mismatch at ({t},{h})")
            expanded=torch.exp2((sf[idx].to(torch.int32)-127).float()).repeat_interleave(BS)
            expected=(decoded[idx]*expanded).to(torch.bfloat16)
            if not torch.equal(bf16[t,h],expected):
                delta=(bf16[t,h].float()-expected.float()).abs().max().item()
                raise AssertionError(f"Engram BF16 mismatch at ({t},{h}) max_abs={delta}")
    print("ENGRAM_SM80_LOOKUP_BYTES_AND_BF16_BITWISE_PASS")

    # The production recipe offloads Engram weights to pinned CPU memory.
    # Test the same UVA device view (without copying its table to HBM).
    # This is intentionally a hard failure: if this op is unavailable in
    # the container, the real --engram-config cpu_offload path also fails.
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    pinned_table=cpu_fp8.pin_memory()
    pinned_scales=sf.cpu().pin_memory()
    if not pinned_table.is_pinned() or not pinned_scales.is_pinned():
        raise RuntimeError("Engram CPU offload requires pinned host memory")
    uva_table=get_accelerator_view_from_cpu_tensor(pinned_table)
    uva_scales=get_accelerator_view_from_cpu_tensor(pinned_scales)
    if uva_table.dtype!=torch.float8_e4m3fn or uva_scales.dtype!=torch.uint8:
        raise AssertionError("Engram UVA view changed FP8 or E8M0 dtype")
    fake.cpu_offload=True
    fake._storage=lambda: (uva_table,uva_scales)
    uva_bf16=torch.empty_like(bf16)
    uva_packed=torch.empty_like(packed)
    ParallelEngramEmbedding.lookup(fake, ids, uva_bf16)
    ParallelEngramEmbedding.lookup(fake, ids, uva_packed)
    torch.cuda.synchronize()
    if not torch.equal(uva_bf16,bf16) or not torch.equal(uva_packed,packed):
        raise AssertionError("CPU pinned UVA Engram values differ from GPU-table reference")
    print("ENGRAM_SM80_PINNED_HOST_UVA_BITWISE_PASS")


if __name__=="__main__":
    main()
