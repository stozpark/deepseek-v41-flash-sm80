#!/usr/bin/env python3
"""Physical A100 Engram FP8 lookup oracle, no model weights.

Compare E4M3FN bytes, UE8M0 scales and BF16 output with an independent
PyTorch reference. Does not validate pinned-host UVA memory or full model.
"""
import importlib.metadata
import torch


def main():
    if importlib.metadata.version("vllm")!="0.32.0":
        raise RuntimeError("Requires pinned vLLM v0.32.0")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)!=(8,0):
        raise RuntimeError("Physical A100/A800 SM80 required")

    from vllm.models.deepseek_v41.common.engram import _engram_lookup_kernel
    torch.manual_seed(62197)
    dev=torch.device("cuda",0)
    T,H,N,D,BS=4,2,32,256,32
    # Produce FP8 on CPU, then transfer raw bytecodes; no CUDA FP8 MMA needed.
    cpu_codes=(torch.randn(N,D,dtype=torch.float32)*0.3).to(
        torch.float8_e4m3fn).view(torch.uint8)
    table=cpu_codes.to(dev).view(torch.float8_e4m3fn).contiguous()
    sf=torch.randint(124,131,(N,D//BS),dtype=torch.uint8).to(dev)
    ids=torch.tensor([[3,19],[0,6],[31,-1],[35,9]],dtype=torch.int64,device=dev)
    bf16=torch.empty((T,H,D),dtype=torch.bfloat16,device=dev)
    packed=torch.empty((T,H,D+D//BS),dtype=torch.uint8,device=dev)
    for result,is_packed in ((bf16,False),(packed,True)):
        _engram_lookup_kernel[(1,)](
            table,sf,ids,ids,result,
            0,N,T*H,ids.stride(0),ids.stride(1),
            HEAD_START=0,LOCAL_HEADS=H,TOTAL_HEADS=H,
            DIM=D,QUANT_BLOCK=BS,BLOCK_R=16,GRID=1,
            SORTED=False,PACKED=is_packed,OUT_STRIDE=result.stride(0),
            SOFTWARE_FP8=True,
        )
    torch.cuda.synchronize()
    decoded=table.float()
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


if __name__=="__main__":
    main()
