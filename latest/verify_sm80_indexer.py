#!/usr/bin/env python3
"""Real A100 sparse-indexer MQA numeric parity and candidate mask probes.

Standalone, no checkpoint. Must run after stage3 port on CUDA13 SM80.
PyTorch FP32/BF16 reference is independent of the Triton kernel.
"""
from __future__ import annotations
import math
import torch
import triton


def assert_close(x, y, label, atol=0.05, rtol=0.02):
    torch.testing.assert_close(x.float(),y.float(),atol=atol,rtol=rtol)
    print(f"{label}_PASS max_abs={(x.float()-y.float()).abs().max().item():.5g}")


def ref_scores(q, k, weights, scales):
    # q:[M,H,D], k:[N,D], scales:[N], weights:[M,H]
    qa=q.to(torch.bfloat16).float()
    ka=k.to(torch.bfloat16).float()
    # The reference follows FP8-dequant -> BF16 GEMM with FP32 accumulation;
    # logits = sum_h weight_h * ReLU(q_h dot k * per-token scale).
    score=torch.einsum("mhd,nd->mhn",qa,ka)*scales.float()[None,None,:]
    return (torch.relu(score)*weights.float()[:,:,None]).sum(dim=1)


def prefill(device):
    from vllm.v1.attention.ops.mqa_logits_triton import fp8_mqa_logits_triton
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,apply_candidate_mask,
    )
    torch.manual_seed(62197)
    M,H,D,N=3,16,128,259
    q=(torch.randn(M,H,D,device=device)*.24).to(torch.float8_e4m3fn)
    k=(torch.randn(N,D,device=device)*.24).to(torch.float8_e4m3fn)
    sc=torch.exp2(torch.randint(-3,3,(N,),device=device).float())
    w=torch.rand(M,H,device=device)*.2
    start=torch.tensor([0,30,114],device=device,dtype=torch.int32)
    end=torch.tensor([259,231,211],device=device,dtype=torch.int32)
    logits=fp8_mqa_logits_triton(q,(k,sc),w,start,end,clean_logits=False)
    ref=ref_scores(q,k,w,sc)
    cols=torch.arange(N,device=device)[None,:]
    ref=ref.masked_fill((cols<start[:,None])|(cols>=end[:,None]),-float("inf"))
    visible=(cols>=start[:,None])&(cols<end[:,None])
    assert torch.isneginf(logits[~visible]).all(), "prefill causal/partial tail not -inf"
    assert_close(logits[visible],ref[visible],"PREFILL_FP8_LOGITS")

    # Candidate block IDs are local to each valid row start.
    # The source selector must pin the newest block, and the consumer must
    # mask all non-candidate blocks identically to a Python oracle.
    block=8
    count=7
    candidates=torch.full((M,count),-1,device=device,dtype=torch.int32)
    select_candidate_blocks(logits.clone(),start,end,count,block,candidates)
    assert ((candidates>=0)|(candidates==-1)).all()
    masked=logits.clone()
    apply_candidate_mask(masked,start,end,candidates,block)
    for i in range(M):
        row=ref[i].clone()
        selected={int(c) for c in candidates[i].tolist() if c>=0}
        for t in range(N):
            if not start[i]<=t<end[i] or (t-int(start[i]))//block not in selected:
                row[t]=-float("inf")
        ref_finite=torch.isfinite(row)
        got_finite=torch.isfinite(masked[i])
        if not torch.equal(ref_finite,got_finite):
            raise AssertionError(f"candidate mask differs on row {i}")
        assert_close(masked[i,ref_finite],row[ref_finite],f"CANDIDATE_ROW_{i}")
    print("TWO_LEVEL_CANDIDATE_MASK_PASS")


def decode(device):
    from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton
    torch.manual_seed(1084)
    B,H,D,BS,NB=2,16,128,128,6
    q=(torch.randn(B,1,H,D,device=device)*.25).to(torch.float8_e4m3fn)
    k=(torch.randn(NB,BS,D,device=device)*.25).to(torch.float8_e4m3fn)
    scales=torch.exp2(torch.randint(-2,3,(NB,BS),device=device).float())
    w=torch.rand(B,H,device=device)*.3
    lens=torch.tensor([253,129],device=device,dtype=torch.int32)
    # Nonmonotone physical block mapping exercises the paged address math.
    table=torch.tensor([[4,1,3],[2,0,5]],device=device,dtype=torch.int32)
    cache=torch.zeros((NB,BS,1,D+4),device=device,dtype=torch.uint8)
    raw=cache.reshape(NB,-1)
    raw[:,:BS*D]=k.view(torch.uint8).reshape(NB,BS*D)
    raw[:,BS*D:]=scales.contiguous().view(torch.uint8).reshape(NB,BS*4)
    result=fp8_paged_mqa_logits_triton(
        q,cache,w,lens,table,max_model_len=280,clean_logits=True)
    assert result.shape==(B,280),result.shape
    for b in range(B):
        n=int(lens[b])
        value=torch.cat([k[int(index)] for index in table[b].tolist()])[:n]
        sf=torch.cat([scales[int(index)] for index in table[b].tolist()])[:n]
        ref=ref_scores(q[b,0][None,:,:],value,w[b:b+1],sf)[0]
        assert_close(result[b,:n],ref,f"PAGED_DECODE_ROW_{b}")
        assert torch.isneginf(result[b,n:]).all(),f"row {b} tail invalid"
        # Nonambiguous Top-K values must match without sorting assumptions.
        ktop=min(16,n)
        actual_set=set(result[b,:n].topk(ktop).indices.tolist())
        ref_set=set(ref.topk(ktop).indices.tolist())
        if actual_set!=ref_set:
            raise AssertionError(f"decode TOP-K mismatch row={b}: {actual_set^ref_set}")
    print("PAGED_DECODE_FP8_LOGITS_AND_TOPK_PASS")


def main():
    import importlib.metadata as md
    if md.version("vllm")!="0.32.0":
        raise RuntimeError("vLLM 0.32.0 required")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)!=(8,0):
        raise RuntimeError("Physical A100/A800 SM80 required")
    device=torch.device("cuda",0)
    prefill(device)
    decode(device)
    torch.cuda.synchronize()
    print("SM80_INDEXER_NUMERIC_PARITY_PASS_NOT_FULL_MODEL_VALIDATION")


if __name__=="__main__":
    main()
