# DeepSeek-V4.1-Flash on A100/A800 (SM80) - CUDA 12.9

Run `deepseek-ai/DeepSeek-V4.1-Flash` on NVIDIA A100/A800 using a SIF built
from the **official vLLM CUDA 12.9 DeepSeek-V4.1 image** plus a pinned Ampere
overlay.

## Base

```text
Official image : vllm/vllm-openai:deepseekv41-flash-0909-cu129
CUDA in image  : 12.9.1
GPU target     : A100/A800 (SM80)
SM80 overlay   : wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36
Output SIF     : deepseek-v41-flash-sm80-cu129.sif
```

The CUDA 13 variant is on `main`.

## Why an overlay is still needed

The official DeepSeek-V4.1 image supplies the model integration and native
CUDA/PyTorch/vLLM stack. A100 still needs an Ampere execution path because
native Triton FP8 E4M3 conversion requires SM89+ and the stock V4.1 sparse-MLA
selection targets newer GPU backends.

The pinned overlay adds:

- SM8x `TRITON_MLA_SPARSE_DSV41` routing
- software E4M3 encode/decode for FP8 cache operations below SM89
- V4.1 128-token indexer pages
- CUDA-capable Triton sparse-MLA path
- pre-SM89 JIT warmup fixes
- DSpark V4.1 hooks from the validated A100 backport pin

The official image's compiled native extensions remain in place. See
[`patches/OVERLAY_MANIFEST.md`](patches/OVERLAY_MANIFEST.md).

## Build the SIF

```bash
git clone https://github.com/stozpark/deepseek-v41-flash-sm80.git
cd deepseek-v41-flash-sm80
git checkout cu129
./build_sif.sh
```

Result:

```text
deepseek-v41-flash-sm80-cu129.sif
```

Use an existing backport checkout if desired:

```bash
BACKPORT_SOURCE=/path/to/vllm-backport ./build_sif.sh
```

Or prepare the overlay once and build without another Git fetch:

```bash
BACKPORT_SOURCE=/path/to/vllm-backport ./prepare_sm80_overlay.sh
SKIP_OVERLAY_PREPARE=1 ./build_sif.sh
```

The generated `patches/overlay/` and `*.sif` are git-ignored.

If fakeroot is unavailable:

```bash
BUILD_ARGS="" ./build_sif.sh
```

## Preflight

```bash
./preflight.sh
```

Recommended starting point:

```text
GPU      : 8 x A100 80GB
Host RAM : 384 GiB+ preferred (256 GiB is tight)
TP       : 8
Context  : 256K first
```

## Serve

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
./serve_tp8_a100.sh
```

Defaults:

```text
TP                         8
max_model_len              262144
max_num_seqs               16
max_num_batched_tokens     16384
gpu_memory_utilization     0.90
KV cache                   fp8_ds_mla
Engram                     CPU offload
Prefix caching             enabled
DSpark                     5 speculative tokens
CUDA graph                 FULL_AND_PIECEWISE
custom all-reduce          disabled
```

Debug startup without speculative decoding:

```bash
DISABLE_DSPARK=1 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

## Verify inside the SIF

```bash
apptainer exec --nv deepseek-v41-flash-sm80-cu129.sif \
  python3 verify_runtime.py
```

This branch is the preferred starting point for A100 systems that cannot run
the CUDA 13 official image directly.
