# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental DeepSeek V4.1 A100/A800 (SM80) sparse MLA backend.

A narrow adaptation of the existing ROCm Triton reference path. This module
only establishes model/backend registration. In particular, this class uses the ROCm BF16 fallback as the correctness
reference; cache, indexer and Engram FP8 have separate architecture patches.
"""
from __future__ import annotations

from vllm.models.deepseek_v41.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterMLAAttention,
)
from vllm.platforms.interface import DeviceCapability


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    """CUDA SM80 V4.1 sparse MLA backend using the portable Triton path."""

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """A100 path; ROCm-specific aiter kernels must never be selected."""

    backend_cls = DeepseekV41AmpereMLASparseBackend

    # Inherit the parent ROCm *BF16/einsum* output projection unchanged.
    # Its _o_proj -> rocm_inv_rope_einsum -> _get_cached_wo_a_bf16 path
    # expects ordinary MXFP8 linear weights with is_bmm=False. The earlier
    # SM80 override is_bmm=True was inconsistent with this inherited path:
    # it selected BMM-specialized post-load quantization without actually
    # dispatching the corresponding CUDA grouped BMM forward. Do not
    # change weight packing until an SM80 BMM forward is implemented
    # and verified against this BF16/einsum reference.
