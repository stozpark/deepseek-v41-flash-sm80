#!/usr/bin/env python3
"""A100 DSpark5 paged MQA causal-mask parity, no model weights needed.

Important: kernel outputs are compared with a separate PyTorch reference.
This test uses physical SM80 and must not be replaced by source-only CI.
"""
import torch


def run():
    import importlib.metadata as md
    if md.version("vllm") != "0.32.0":
        raise RuntimeError("Only pinned vLLM 0.32.0 supported")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 0):
        raise RuntimeError("Physical A100/A800 SM80 required")
    from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton
    torch.manual_seed(171017)
    device="cuda"
    B, Q, H, D, BS, NBL = 2, 6, 16, 128, 128, 8
    q=(torch.randn(B,Q,H,D,device=device)*0.25).to(torch.float8_e4m3fn)
    keys=(torch.randn(NBL,BS,D,device=device)*0.25).to(torch.float8_e4m3fn)
    sf=torch.exp2(torch.randint(-2,3,(NBL,BS),device=device).float()).contiguous()
    weights=torch.rand(B*Q,H,device=device).contiguous()
    cache=torch.zeros((NBL,BS,1,D+4),device=device,dtype=torch.uint8)
    raw=cache.reshape(NBL,-1)
    raw[:,:BS*D]=keys.view(torch.uint8).reshape(NBL,-1)
    raw[:,BS*D:]=sf.view(torch.uint8).reshape(NBL,-1)
    table=torch.tensor([[7,0,3,4],[5,2,1,6]],device=device,dtype=torch.int32)
    # This MUST be (B,Q): same request, six distinct causal ends.
    lens=torch.tensor([[130,131,132,133,134,135],
                       [250,251,252,253,254,255]],device=device,dtype=torch.int32)
    logits=fp8_paged_mqa_logits_triton(q,cache,weights,lens,table,
                                      max_model_len=260,clean_logits=True)
    assert logits.shape == (B*Q,260)
    for b in range(B):
        all_keys=torch.cat([keys[int(v)] for v in table[b].tolist()]).float()
        all_sc=torch.cat([sf[int(v)] for v in table[b].tolist()]).float()
        for qi in range(Q):
            n=int(lens[b,qi])
            row=b*Q+qi
            ref=torch.einsum("hd,nd->hn",q[b,qi].float().to(torch.bfloat16).float(),
                             all_keys[:n].to(torch.bfloat16).float())
            ref=(torch.relu(ref*all_sc[:n][None,:])*weights[row,:,None]).sum(dim=0)
            torch.testing.assert_close(logits[row,:n],ref,rtol=0.02,atol=0.06)
            if not torch.isneginf(logits[row,n:]).all():
                raise AssertionError(f"DSpark causal tail nonmasked: b={b},q={qi},n={n}")
    print("SM80_DSPARK5_MULTITOKEN_CAUSAL_PARITY_PASS")


if __name__=="__main__":
    run()
