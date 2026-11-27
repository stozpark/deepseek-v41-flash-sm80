"""Fail closed if the actual loaded HYV4 layer dispatch differs from the plan."""
import os
import torch
from vllm.v1.worker.gpu_worker import Worker
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.models.hy_v4.nvidia.attention import HYV4MLAAttention
from vllm.models.hy_v4.nvidia.sm80_sparse import HYV4SM80SparseImpl
from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExpertsBase
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend


def require(ok, message):
    if not ok:
        raise RuntimeError("HY4 dispatch audit: "+message)


class AuditedWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.get_model()
        cfg = self.vllm_config.model_config.hf_text_config
        fixture = os.environ.get("HY4_FIXTURE_MODE") == "1"
        require(fixture or cfg.num_hidden_layers==78, "not full flagship")
        pp = get_pp_group()
        tp = get_tp_group()
        text = getattr(model,"language_model",model)
        backbone = text.model
        start,end = backbone.start_layer,backbone.end_layer
        require(end>start, "empty stage")
        require(self.vllm_config.attention_config.sparse_mla_force_mqa, "dense prefill allowed")
        require(not self.vllm_config.attention_config.sparse_mla_shard_q, "ShardQ enabled")
        quantized = self.vllm_config.quant_config is not None
        if not fixture:
            require(quantized, "full flagship requires quantization")
        layers = [m for m in model.modules() if isinstance(m,HYV4MLAAttention)]
        require(sorted(m.layer_id for m in layers)==list(range(start,end)), "missing or unexpected attention layers")
        for m in layers:
            impl = m.mla_attn.impl
            require(type(impl) is HYV4SM80SparseImpl, f"layer {m.layer_id}: {type(impl)}")
            require(impl.sinks.data_ptr()==m.learnable_sink_param.data_ptr(),
                    f"layer {m.layer_id}: sink alias mismatch")
            require(torch.isfinite(impl.sinks).all().item(), "nonfinite sink")
            require(m.gated_mla and m.linear_gate is not None and not m.use_hpc_gated_mla,
                    f"layer {m.layer_id}: wrong gate path")
            require((m.indexer is not None)==(cfg.indexer_types[m.layer_id]=="full"),
                    f"layer {m.layer_id}: wrong IndexCache pattern")
            if quantized and m.layer_id!=0:
                for projection in (m.o_proj,m.linear_gate):
                    kernel = getattr(getattr(projection,"scheme",None),"linear_kernel",None)
                    require(isinstance(kernel,MarlinFP8ScaledMMLinearKernel) and kernel.block_quant,
                            f"layer {m.layer_id}: FP8 linear not block Marlin: {type(kernel)}")
                    require(projection.weight.dtype==torch.int32, "unpacked FP8 linear")
        experts = [m for m in model.modules() if isinstance(m,RoutedExperts)]
        expected = sum(t=="sparse" for t in cfg.mlp_layer_types[start:end])
        require(len(experts)==expected,f"MoE count {len(experts)} != {expected}")
        if quantized:
            for m in experts:
                method = m.quant_method
                require(getattr(method,"fp8_backend",None)==Fp8MoeBackend.MARLIN, "wrong FP8 MoE oracle")
                kernel = getattr(method,"moe_kernel",None)
                require(kernel is not None and isinstance(kernel.fused_experts,MarlinExpertsBase),
                        "wrong actual MoE experts")
                require(m.w13_weight.dtype==torch.int32 and m.w2_weight.dtype==torch.int32,
                        "unpacked FP8 experts")
                require(m.swiglu_limit==10.0 and kernel.fused_experts.activation_config.clamp_limit==10.0,
                        "SwiGLU clamp lost")
        print(f"HY4_MODEL_DISPATCH=PASS pp={pp.rank_in_group} tp={tp.rank_in_group} "
              f"layers={start}:{end} attention={len(layers)} moe={len(experts)} fp8={quantized}",
              flush=True)
        # Eager-only first-call instrumentation. The fixture/API gate must
        # observe BOTH prefill and decode on every rank, not merely the load.
        if os.environ.get("HY4_AUDIT_TRACE")=="1":
            for m in layers:
                self._trace_sparse(m,pp.rank_in_group,tp.rank_in_group)

    @staticmethod
    def _trace_sparse(m,pp,tp):
        impl = m.mla_attn.impl
        old = impl._forward_mqa_kernel
        seen=set()
        def wrapped(*a,**kw):
            result=old(*a,**kw)
            phase="decode" if kw["is_decode"] else "prefill"
            if phase not in seen:
                seen.add(phase)
                print(f"HY4_SPARSE_FORWARD=PASS pp={pp} tp={tp} layer={m.layer_id} phase={phase}",flush=True)
            return result
        impl._forward_mqa_kernel=wrapped
