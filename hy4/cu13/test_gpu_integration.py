"""Real indexer cache -> score -> selection -> sparse MLA plus large strides.

These exercise production kernels with synthetic data, not the full model.
Model-level dispatch and forward coverage are separately enforced by the worker
and fixture/API probes. No GPU skip is counted as a pass.
"""
import torch
from vllm import _custom_ops as ops
from vllm.model_executor.layers.indexer_topk import get_indexer_topk
from vllm.models.hy_v4.nvidia.sm80_indexer_quant import quantize_indexer_q
from vllm.models.hy_v4.nvidia.sink_math import apply_attention_sink
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view, triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton, fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd
from reference_checks import check_topk


def test_indexer_chain():
    torch.manual_seed(92)
    page, d, heads, topk = 64, 128, 32, 2048
    lengths = [2209, 33]
    nblocks = 36
    order = torch.randperm(nblocks, dtype=torch.int64)
    table = torch.zeros((2, 35), dtype=torch.int32)
    table[0] = order[:35].int()
    table[1, 0] = order[35].int()
    slots = torch.cat([table[r, torch.arange(n)//page].long()*page
                       + torch.arange(n)%page for r, n in enumerate(lengths)])
    total = sum(lengths)
    k = torch.randn(total, d, dtype=torch.bfloat16)
    backing = torch.zeros((nblocks, page*(d+4)+32), dtype=torch.uint8, device="cuda")
    cache = backing[:, :-32].view(nblocks, page, d+4)
    ops.indexer_k_quant_and_cache(k.cuda(), cache, slots.cuda(), d, "ue8m0")
    # Use the actual native gather and verify bytes/scales, not a synthetic
    # cache assembled from the same reference used by the scoring test.
    gathered = torch.empty(total, d, device="cuda", dtype=torch.uint8)
    scale_bytes = torch.empty(total, 4, device="cuda", dtype=torch.uint8)
    cu = torch.tensor([0, lengths[0], total], dtype=torch.int32, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(cache, gathered, scale_bytes, table.cuda(), cu)
    torch.cuda.synchronize()
    ks = torch.exp2(torch.ceil(torch.log2(k.float().abs().amax(-1).clamp_min(1e-4)/448)))
    k8 = (k.float()/ks[:, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(scale_bytes.cpu().view(torch.float32).flatten(), ks, atol=0, rtol=0)
    assert torch.equal(gathered.cpu(), k8.view(torch.uint8))
    q = torch.randn(2, heads, d, dtype=torch.bfloat16)
    q8, qs = quantize_indexer_q(q.reshape(-1, d).cuda(), d, True)
    q8 = q8.reshape(2, heads, d)
    raw_w = torch.randn(2, heads)
    w = raw_w.cuda() * qs.reshape(2, heads) * (d*heads)**-0.5
    ref_qs = torch.exp2(torch.ceil(torch.log2(q.float().abs().amax(-1).clamp_min(1e-10)/448)))
    ref_q8 = (q.float()/ref_qs[..., None]).clamp(-448,448).to(torch.float8_e4m3fn)
    torch.testing.assert_close(qs.cpu().reshape(2,heads), ref_qs, atol=0, rtol=0)
    assert torch.equal(q8.view(torch.uint8).cpu(), ref_q8.view(torch.uint8))
    ref_w = raw_w * ref_qs * (d*heads)**-0.5
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    pre = fp8_mqa_logits_triton(q8, (gathered.view(torch.float8_e4m3fn),
                                     scale_bytes.view(torch.float32).flatten()),
                                w, cu[:-1], cu[1:], lengths[0])
    # Use page-aligned width for persistent Top-K.
    dec = fp8_paged_mqa_logits_triton(q8[:,None], cache, w, lens[:,None],
                                      table.cuda(), 2240)
    # The actual auto decode selector chooses persistent_topk on SM80,
    # which requires the per-device WorkspaceManager used by GPUModelRunner.
    from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager
    selector = get_indexer_topk("auto")
    pre_ids = torch.full((2,topk), -1, device="cuda", dtype=torch.int32)
    dec_ids = torch.full_like(pre_ids, -1)
    init_workspace_manager(torch.device("cuda", torch.cuda.current_device()))
    try:
        selector.forward_prefill(pre, None, lens, pre_ids, topk)
        selector(dec, lens[:,None], 1, dec_ids, topk, max(lengths))
        torch.cuda.synchronize()
    finally:
        reset_workspace_manager()
    refs = []
    for r,n in enumerate(lengths):
        offset = 0 if r == 0 else lengths[0]
        keys = k8[offset:offset+n].float()
        ref = ((ref_q8[r].float() @ keys.T).relu()*ref_w[r,:,None]).sum(0)*ks[offset:offset+n]
        refs.append(ref)
        for scores, ids in ((pre,pre_ids),(dec,dec_ids)):
            torch.testing.assert_close(scores[r,:n].cpu(), ref, atol=0.002, rtol=0.002)
            check_topk(ids[r].cpu().tolist(), ref.tolist(), n, topk,
                       tol=0.002+0.002*float(ref.abs().max()))
            # Independent selection against the actual computed logits, too.
            check_topk(ids[r].cpu().tolist(), scores[r,:n].cpu().tolist(), n, topk,
                       tol=1e-6)
    # Full physical-index conversion including inter-layer page gaps.
    stride_rows = 80
    latent = torch.randn(nblocks, stride_rows, 576, dtype=torch.bfloat16, device="cuda")
    kv = latent[:, :page]
    flat, stride = flat_kv_row_view(kv, page)
    assert stride == stride_rows
    phys, counts = triton_convert_req_index_to_global_index(
        torch.arange(2, device="cuda", dtype=torch.int32), table.cuda(), dec_ids,
        BLOCK_SIZE=page, BLOCK_STRIDE_ROWS=stride, NUM_TOPK_TOKENS=topk,
        return_valid_counts=True,
    )
    expected_ids = dec_ids.cpu().clone()
    for r in range(2):
        valid = expected_ids[r] >= 0
        a = expected_ids[r,valid].long()
        expected_ids[r,valid] = table[r,a//page]*stride + (a%page).int()
    assert torch.equal(phys.cpu(), expected_ids)
    assert counts.cpu().tolist() == [2048,33]
    aq = torch.randn(2,8,576,device="cuda",dtype=torch.bfloat16)*0.25
    sink = torch.linspace(-3,3,8,device="cuda",dtype=torch.float32)
    raw,lse = triton_mla_sparse_fwd(aq,flat,phys,sm_scale=256**-0.5,
                                    kv_lora_rank=512,return_lse=True)
    out,_ = apply_attention_sink(raw,lse,sink)
    for r in range(2):
        ids = expected_ids[r][expected_ids[r]>=0].long().cuda()
        keys = flat[ids].float().cpu()
        logits = aq[r].float().cpu() @ keys.T * (256**-0.5)
        probs = torch.cat((logits,sink.cpu()[:,None]),-1).softmax(-1)[:,:-1]
        expected = probs @ keys[:,:512]
        torch.testing.assert_close(out[r].float().cpu(),expected,atol=0.005,rtol=0.03)
    print("HY4_INDEXER_CHAIN=PASS")


def test_large_addresses():
    # Touch two sparse pages in a >4GiB allocation; do NOT zero the entire buffer.
    # Check the exact high page plus an address near the 2GiB boundary.
    d, page = 576, 64
    stride = ((1<<31)//(2*d)) + 64
    buf = torch.empty((2*stride+page,d),device="cuda",dtype=torch.bfloat16)
    kv = buf.as_strided((3,page,d),(stride*d,d,1))
    for p in range(3):
        kv[p].fill_(p+1)
    flat, step = flat_kv_row_view(kv,page)
    assert step == stride and 2*step*d*2 > 1<<32
    q = torch.zeros((1,8,d),device="cuda",dtype=torch.bfloat16)
    idx = torch.full((1,2048),-1,device="cuda",dtype=torch.int32)
    idx[0,:3] = torch.tensor([0,stride,2*stride],device="cuda",dtype=torch.int32)
    for splits in (1,4):
        out,lse = triton_mla_sparse_fwd(q,flat,idx,sm_scale=256**-0.5,
                                        kv_lora_rank=512,return_lse=True,num_kv_splits=splits)
        torch.testing.assert_close(out.float(),torch.full_like(out.float(),2),atol=0.01,rtol=0)
        torch.testing.assert_close(lse,torch.full_like(lse,1.0986123),atol=1e-5,rtol=0)
    del buf,kv,flat,out,lse
    torch.cuda.empty_cache()
    d = 128
    stride_bytes = (1<<31)+page*(d+4)
    backing = torch.empty(2*stride_bytes+page*(d+4),device="cuda",dtype=torch.uint8)
    cache = backing.as_strided((3,page,d+4),(stride_bytes,d+4,1))
    k = torch.tensor([1.,2.,4.],dtype=torch.bfloat16)[:,None].expand(3,d).contiguous()
    slots = torch.tensor([0,page,2*page],device="cuda",dtype=torch.int64)
    ops.indexer_k_quant_and_cache(k.cuda(),cache,slots,d,"ue8m0")
    table = torch.tensor([[2],[1],[0]],device="cuda",dtype=torch.int32)
    cu = torch.arange(4,device="cuda",dtype=torch.int32)
    g = torch.empty((3,d),device="cuda",dtype=torch.uint8)
    s = torch.empty((3,4),device="cuda",dtype=torch.uint8)
    ops.cp_gather_indexer_k_quant_cache(cache,g,s,table,cu)
    got = g.cpu().view(torch.float8_e4m3fn).float()*s.cpu().view(torch.float32)
    torch.testing.assert_close(got,k.flip(0).float(),atol=0,rtol=0)
    q8 = torch.ones(3,1,32,d,dtype=torch.float32).to(torch.float8_e4m3fn)
    q8 = q8.view(torch.uint8).cuda().view(torch.float8_e4m3fn)
    logits = fp8_paged_mqa_logits_triton(q8,cache,torch.ones(3,32,device="cuda"),
                  torch.ones((3,1),device="cuda",dtype=torch.int32),table,64)
    torch.testing.assert_close(logits[:,0].cpu(),torch.tensor([4.,2.,1.])*d*32,atol=0,rtol=0)
    torch.cuda.synchronize()
    print("HY4_ADDRESS_2G_4G=PASS")
