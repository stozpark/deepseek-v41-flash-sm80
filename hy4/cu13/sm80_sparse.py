# SPDX-License-Identifier: Apache-2.0
"""HYV4 sink-aware SM80 sparse MLA, atop the pinned #68633 Triton backend.

Design reference: vLLM #69779 (Flash/Ada); no Flash-specific IndexPool or SWA.
"""
from typing import Any
import torch

from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend, TritonMLASparseImpl,
)
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd


from .sink_math import apply_attention_sink


class HYV4SM80SparseImpl(TritonMLASparseImpl):
    supports_dense_mha_prefill = False
    supports_shardq = False

    def __init__(self, *args: Any, sinks: torch.Tensor, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if sinks.dtype != torch.float32 or sinks.shape != (self.num_heads,):
            raise ValueError("HYV4 SM80 requires one FP32 sink per local head")
        if self.kv_lora_rank != 512 or self.qk_rope_head_dim != 64:
            raise ValueError("HYV4 SM80 requires latent512 + RoPE64")
        if self.dcp_world_size != 1 or self.pcp_world_size != 1:
            raise NotImplementedError("HYV4 SM80 context parallelism is not validated")
        self.sinks = sinks

    def _forward_mqa_kernel(
        self, q, kv_cache, topk_indices, valid_counts, *, layer, block_size,
        is_decode,
    ):
        q_tensor = torch.cat(q, dim=-1) if isinstance(q, tuple) else q
        kv_rows, _ = flat_kv_row_view(kv_cache, block_size)
        out, lse = triton_mla_sparse_fwd(
            q_tensor, kv_rows, topk_indices,
            sm_scale=self.scale, kv_lora_rank=self.kv_lora_rank,
            return_lse=True,
        )
        if lse is None:
            raise RuntimeError("SM80 sparse MLA did not return LSE")
        # CP is forbidden; upstream real-key LSE convention is not altered.
        # Do not spend a second logaddexp launch when LSE is not consumed.
        if self.need_to_return_lse_for_decode:
            raise RuntimeError("unexpected LSE consumer on HYV4 SM80")
        out, _ = apply_attention_sink(out, lse, self.sinks, return_lse=False)
        return out, None


class HYV4SM80SparseBackend(TritonMLASparseBackend):
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(8, 0)

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @classmethod
    def supports_dcp(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls():
        return HYV4SM80SparseImpl
