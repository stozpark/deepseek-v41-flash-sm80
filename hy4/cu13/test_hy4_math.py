"""GPU checks of the unchanged upstream Hy4-specific iHC and M-RoPE paths."""
import torch
from vllm.models.hy_v4.nvidia.triton_ihc import (
    triton_ihc_pre, triton_ihc_post, triton_ihc_supported,
)
from vllm.models.hy_v4.nvidia.mrope import set_token_rope, token_rope, rope_into, apply_mrope
from vllm.model_executor.layers.rotary_embedding import get_rope


def test_ihc():
    torch.manual_seed(410)
    for t in (1,17):
        x=torch.randn(t,4,6144,dtype=torch.bfloat16)
        weight=torch.randn(8,4*6144,dtype=torch.float32)*0.005
        scale=torch.tensor([0.7,1.3],dtype=torch.float32)
        base=torch.linspace(-2,2,8,dtype=torch.float32)
        gx=x.cuda()
        assert triton_ihc_supported(gx)
        got,post=triton_ihc_pre(gx,weight.cuda(),scale.cuda(),base.cuda(),2.,1e-6,1e-5)
        flat=x.float().flatten(1)
        mixes=(flat@weight.T)*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-5)
        pre=(mixes[:,:4]*scale[0]+base[:4]).sigmoid()+1e-6
        ref_post=2*(mixes[:,4:]*scale[1]+base[4:]).sigmoid()+1e-6
        ref=(pre[...,None]*x.float()).sum(1)
        torch.testing.assert_close(got.float().cpu(),ref,atol=0.02,rtol=0.01)
        torch.testing.assert_close(post.cpu(),ref_post,atol=2e-5,rtol=2e-5)
        sub=torch.randn(t,6144,dtype=torch.bfloat16)
        # Model's gate operation is elementwise sigmoid after projection;
        # compare its BF16 result followed by iHC post against FP32 math.
        gate=torch.randn_like(sub)
        gated=sub.cuda()*torch.sigmoid(gate.cuda())
        ref_gated=(sub.float()*gate.float().sigmoid().bfloat16().float()).bfloat16()
        torch.testing.assert_close(gated.float().cpu(),ref_gated.float(),atol=0.015,rtol=0.01)
        actual=triton_ihc_post(gated,gx,post)
        ref_out=(ref_post[...,None]*ref_gated.float()[:,None,:]+x.float()).bfloat16()
        torch.testing.assert_close(actual.float().cpu(),ref_out.float(),atol=0.04,rtol=0.02)
    print("HY4_IHC_GATE=PASS")


def rotate(x,rows):
    c,s=rows[:,:32,None].transpose(1,2).float(),rows[:,32:,None].transpose(1,2).float()
    x=x.float()
    a,b=x[...,::2],x[...,1::2]
    return torch.stack((a*c-b*s,b*c+a*s),-1).flatten(-2).bfloat16()


def test_mrope():
    torch.manual_seed(413)
    with torch.device("cuda"):
        rope=get_rope(64,8192,is_neox_style=False,rope_parameters={
            "rope_type":"default","rope_theta":10000000,
            "mrope_section":[12,10,10],"mrope_interleaved":True,
        },dtype=torch.bfloat16)
    q=torch.randn(7,8,128,dtype=torch.bfloat16,device="cuda")
    k=torch.randn(7,1,128,dtype=torch.bfloat16,device="cuda")
    q0,k0=q.cpu(),k.cpu()
    for positions in (
        torch.arange(7,device="cuda"),
        torch.tensor([[1,2,3,4,5,6,7],[9,8,7,6,5,4,3],[17,18,19,20,21,22,23]],
                     dtype=torch.int64,device="cuda"),
    ):
        set_token_rope(rope,positions,q)
        cache=rope._match_cos_sin_cache_dtype(q)
        pos,rows=token_rope(rope,positions,cache)
        channels=torch.arange(32)
        p=positions.cpu()
        if p.ndim==1:
            expected=cache.cpu()[p]
        else:
            # Independent per-frequency T/H/W selector, not the vLLM helper.
            which=torch.zeros(32,dtype=torch.long)
            which[(channels%3==1)&(channels<30)]=1
            which[(channels%3==2)&(channels<30)]=2
            base_cache=cache.cpu()
            expected=torch.stack([torch.cat((base_cache[p[which,i],channels],
                                   base_cache[p[which,i],32+channels])) for i in range(7)])
        torch.testing.assert_close(rows[pos].cpu(),expected,atol=0,rtol=0)
        qr,kr=q0.cuda(),k0.cuda()
        # Real Indexer fallback rotates only the trailing 64 channels.
        if positions.ndim==1:
            a,b=rope(positions,qr[...,64:],kr[...,64:])
            qr[...,64:]=a; kr[...,64:]=b
        else:
            apply_mrope(rope,positions,qr[...,64:],kr[...,64:])
        torch.testing.assert_close(qr[...,64:].float().cpu(),rotate(q0[...,64:],expected).float(),
                                   atol=0.015,rtol=0.01)
        torch.testing.assert_close(kr[...,64:].float().cpu(),rotate(k0[...,64:],expected).float(),
                                   atol=0.015,rtol=0.01)
        assert torch.equal(qr[...,:64].cpu(),q0[...,:64])
        assert torch.equal(kr[...,:64].cpu(),k0[...,:64])
        # Also validate the main-attention fused path's strided write.
        dst=q0.cuda()
        rope_into(pos,q[...,64:],rows,dst[...,64:])
        torch.testing.assert_close(dst[...,64:].float().cpu(),rotate(q0[...,64:],expected).float(),
                                   atol=0.015,rtol=0.01)
        assert torch.equal(dst[...,:64].cpu(),q0[...,:64])
    print("HY4_MROPE_TRAILING64=PASS")
