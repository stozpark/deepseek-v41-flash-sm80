#!/usr/bin/env python3
"""Exercise pinned FP8 block repack + Marlin linear and MoE with CPU references.

Uses actual TP8 local matrix dimensions and selected expert subset. This is
not a model-loader or distributed test; the audited worker/fixture covers
dispatch and PP, and matched-checkpoint logits remain the final gate.
"""
import torch
import torch.nn.functional as F
from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import FP8ScaledMMLinearLayerConfig
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym, kFp8DynamicTensorSym,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import prepare_fp8_moe_layer_for_marlin
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig, MoEActivation
from vllm.scalar_type import scalar_types


def cuda_fp8(cpu):
    return cpu.view(torch.uint8).cuda().view(torch.float8_e4m3fn)


def block_weight(shape, factor):
    # Distinct nonuniform 128x128 scales detect transposition/packing bugs.
    scales = (torch.rand((*shape[:-2],shape[-2]//128,shape[-1]//128)) + 0.5)*factor
    w = torch.randn(shape).clamp(-8,8).to(torch.float8_e4m3fn)
    deq = w.float()*scales.bfloat16().float().repeat_interleave(128,-2).repeat_interleave(128,-1)
    return w,scales,deq


def relative_check(actual, expected, limit, label):
    a = actual.float().cpu()
    b = expected.float().cpu()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or b.norm()==0:
        raise AssertionError(label+" invalid output/reference")
    error = float((a-b).norm()/b.norm())
    if error > limit:
        raise AssertionError(f"{label}: relative L2={error:.6f} > {limit}")
    print(label, "relative_L2",error)


@torch.inference_mode()
def test_linear():
    torch.manual_seed(113)
    for n,k in ((6144,2048),(2048,6144)):
        q,s,ref = block_weight((n,k),0.03)
        layer = torch.nn.Module()
        layer.input_size_per_partition = k
        layer.output_size_per_partition = n
        layer.weight_block_size = [128,128]
        layer.orig_dtype = torch.bfloat16
        layer.weight = torch.nn.Parameter(cuda_fp8(q),requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(s.cuda(),requires_grad=False)
        # Exact compressed-tensors block alias used by the production scheme.
        layer.weight_scale_inv = layer.weight_scale
        kernel = MarlinFP8ScaledMMLinearKernel(
            FP8ScaledMMLinearLayerConfig(weight_quant_key=kFp8Static128BlockSym,
              activation_quant_key=kFp8DynamicTensorSym,weight_shape=(n,k),
              input_dtype=torch.bfloat16,out_dtype=torch.bfloat16),
            ["weight","weight_scale","input_scale","input_scale_ub"],
        )
        kernel.process_weights_after_loading(layer)
        assert layer.weight.dtype == torch.int32
        assert layer.weight_scale_inv.data_ptr() == layer.weight_scale.data_ptr()
        assert layer.weight_scale.ndim == 2 and layer.weight_scale.numel() > s.numel()
        for m in (1,17):
            x = torch.randn(m,k,dtype=torch.bfloat16)
            actual = kernel.apply_weights(layer,x.cuda())
            expected = x.float() @ ref.T
            relative_check(actual,expected,0.02,f"MARLIN_LINEAR_{n}_{k}_{m}")


@torch.inference_mode()
def test_moe():
    torch.manual_seed(117)
    e,k,n,m = 16,6144,256,3  # full TP8 local K/N, subset of routed experts
    q1,s1,r1 = block_weight((e,2*n,k),0.10)
    q2,s2,r2 = block_weight((e,k,n),0.02)
    layer = torch.nn.Module()
    layer.num_experts=e
    layer.hidden_size=k
    layer.intermediate_size_per_partition=n
    layer.orig_dtype=torch.bfloat16
    layer.weight_block_size=[128,128]
    w1,w2,z1,z2 = prepare_fp8_moe_layer_for_marlin(
        layer,cuda_fp8(q1),cuda_fp8(q2),s1.cuda(),s2.cuda())
    x = torch.randn(m,k,dtype=torch.bfloat16)*2
    ids = torch.stack([torch.randperm(e)[:8] for _ in range(m)]).int()
    weights = torch.rand(m,8)
    weights = weights/weights.sum(-1,keepdim=True)
    actual = fused_marlin_moe(
        x.cuda(),w1,w2,None,None,z1,z2,weights.cuda(),ids.cuda(),
        scalar_types.float8_e4m3fn.id,activation=MoEActivation.SILU,
        activation_config=ApplyMoEActivationConfig(clamp_limit=10.0))
    ref = torch.zeros(m,k)
    unclamped = torch.zeros_like(ref)
    for i in range(m):
        for j,ex in enumerate(ids[i].tolist()):
            gate,up = (r1[ex]@x[i].float()).bfloat16().float().chunk(2)
            h = (F.silu(gate.clamp(max=10))*up.clamp(-10,10)).bfloat16().float()
            ref[i] += weights[i,j]*(r2[ex]@h)
            unclamped[i] += weights[i,j]*(r2[ex]@(F.silu(gate)*up))
    assert (ref-unclamped).norm()/ref.norm() > 0.1, "fixture failed to exercise clamp"
    relative_check(actual,ref,0.04,"MARLIN_MOE_TOP8_CLAMP")


if __name__=="__main__":
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()!=(8,0):
        raise RuntimeError("physical SM80 required")
    test_linear()
    test_moe()
    torch.cuda.synchronize()
    print("HY4_MARLIN_NUMERICS=PASS")
