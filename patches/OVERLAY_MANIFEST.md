# SM80 overlay

The final SIF is based on the official vLLM DeepSeek-V4.1 image. Only the
Python/Triton pieces required for Ampere are overlaid from the pinned
`wtdcode/vllm-backport` commit.

Vendored at build time:

- `vllm/models/deepseek_v4/`
- `vllm/models/deepseek_v4_1/`
- `vllm/model_executor/layers/sparse_attn_indexer.py`
- `vllm/model_executor/warmup/cutedsl_warmup.py`
- `vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/v1/attention/backends/mla/sparse_swa.py`
- `vllm/v1/attention/ops/common.py`
- `vllm/v1/attention/ops/fp8_sm80.py`
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

The official image's CUDA/PyTorch stack and compiled native extensions remain
in place. `apply_overlay.py` refuses to patch an image that lacks the expected
DeepSeek-V4.1, MXFP4 Marlin and MXFP8 emulation baseline files.
