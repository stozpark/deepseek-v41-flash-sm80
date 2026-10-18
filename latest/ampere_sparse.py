# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental DeepSeek V4.1 A100/A800 (SM80) sparse MLA backend.

A narrow adaptation of the existing ROCm Triton reference path. This module
only establishes model/backend registration. In particular, cache, indexer
and Engram FP8 paths still require a separate architecture-correct port.
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The ROCm class turns this off for its BF16 einsum path. CUDA Marlin
        # repacks the ordinary MXFP8 linear weight, so grouped-BMM uses the
        # MXFP8 emulation kernel instead (same observation as the old SM80
        # DeepSeek V4.1 backport). Further GPU numerical checks are mandatory.
        self.wo_a.is_bmm = True
