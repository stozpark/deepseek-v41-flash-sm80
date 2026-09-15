# SM80 patch layout

## Production path (`main`, CUDA 13)

The production SIF applies only:

- `patches/generated/sm80-cu130-minimal.patch`
- `patches/generated/sm80-cu130-minimal.patch.sha256`
- `patches/generated/sm80-cu130-minimal.manifest.txt`
- `patches/apply_minimal_patch.py`

The verified production delta contains exactly **25 changed files** and is replayed against the pristine official image `vllm/vllm-openai:deepseekv41-flash-0909`.

It excludes model backends under:

- `models/deepseek_v4/amd/`
- `models/deepseek_v4/cpu/`
- `models/deepseek_v4/xpu/`
- `models/deepseek_v4_1/amd/`

Official DeepSeek-V4.1 vision/VL preprocessing files remain from the official image unchanged.

`v1/attention/ops/rocm_aiter_mla_sparse.py` remains intentionally in the minimal patch: despite the filename, the backport reuses its portable Triton sparse-MLA implementation on CUDA SM80.

## Audit/regeneration path

`patches/vendor/` is a pinned broad source snapshot from:

`wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36`

It is retained to regenerate and audit the patch, but is **not copied into the production SIF**. `patches/apply_overlay.py` is audit/regeneration tooling; production uses `apply_minimal_patch.py`.
