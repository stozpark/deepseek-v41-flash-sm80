# SPDX-License-Identifier: Apache-2.0
"""Experimental Hy4 flagship sink-aware BF16 sparse MLA adapter for SM80+.

Based on the API at vLLM PR #68633 head 3470b0db.
The generic backend remains unchanged; only HYV4 resolves to this subclass.
"""
from typing import ClassVar

import torch

from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.sparse_utils import flat_kv_row_view
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseImpl,
)
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd

from .sm80_sink_math import apply_sink


class HYV4TritonMLASparseImpl(TritonMLASparseImpl):
    # ShardQ would require gathering full-head sinks; not supported here.
    supports_shardq: ClassVar[bool] = False

    def __init__(self, *args, sinks: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if sinks is None or sinks.dtype != torch.float32:
            raise ValueError("HYV4 requires FP32 attention sinks")
        if sinks.ndim != 1 or sinks.shape[0] != self.num_heads:
            raise ValueError("HYV4 sink must match local TP heads")
        self.sinks = sinks
        if self.dcp_world_size != 1:
            raise NotImplementedError("HYV4 SM80 sink path does not support DCP")

    def _forward_mqa_kernel(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        valid_counts: torch.Tensor,
        *,
        layer: AttentionLayer,
        block_size: int,
        is_decode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # The pinned generic Triton backend uses physical indices with -1
        # sentinel masking, not a prefix-only valid_counts mask.
        if isinstance(q, tuple):
            q_nope, q_rope = q
            q_tensor = (
                q_nope if self.qk_rope_head_dim == 0
                else torch.cat((q_nope, q_rope), dim=-1)
            )
        else:
            q_tensor = q
        kv_rows, _ = flat_kv_row_view(kv_cache, block_size)
        out, lse = triton_mla_sparse_fwd(
            q_tensor, kv_rows, topk_indices, sm_scale=self.scale,
            kv_lora_rank=self.kv_lora_rank, return_lse=True,
        )
        if lse is None:
            raise RuntimeError("Hy4 sink requires real-key natural-log LSE")
        # Apply exactly once after split-KV reduction. Keep the original
        # real-key LSE convention of FlashMLA; DCP is forbidden above.
        out = apply_sink(out, lse, self.sinks)
        return out, lse if self.need_to_return_lse_for_decode else None


class HYV4TritonMLASparseBackend(TritonMLASparseBackend):
    # Inherit name, metadata, builder, dtype and capability contracts.
    @staticmethod
    def get_impl_cls() -> type[HYV4TritonMLASparseImpl]:
        return HYV4TritonMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
