# SM80 vendored overlay

This repository is designed for disconnected/offline build hosts.

The A100/SM80 runtime sources are committed under `patches/vendor/`; SIF build
does **not** clone GitHub, fetch a commit, use curl, or download patch files.
The vendor snapshot is generated from:

- source: `https://github.com/wtdcode/vllm-backport`
- commit: `24cb31bb4fd0becee65c810c913a8caa4f610c36`

Vendored runtime scope:

- complete `vllm/models/deepseek_v4/`
- complete `vllm/models/deepseek_v4_1/`
- `vllm/model_executor/layers/sparse_attn_indexer.py`
- `vllm/model_executor/warmup/cutedsl_warmup.py`
- `vllm/model_executor/warmup/flashinfer_sparse_mla_warmup.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/v1/attention/backends/mla/sparse_swa.py`
- `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`
- `vllm/v1/attention/ops/common.py`
- `vllm/v1/attention/ops/fp8_sm80.py`
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`

`patches/vendor/BACKPORT_COMMIT` records the source pin and
`patches/vendor/SHA256SUMS` verifies every vendored `vllm/` file before build.

`.github/workflows/vendor-sm80.yml` is only a maintainer refresh mechanism. It
is not used on the offline build host.
