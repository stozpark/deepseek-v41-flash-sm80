# DeepSeek-V4.1-Flash on A100/A800 (SM80) - CUDA 13

Run `deepseek-ai/DeepSeek-V4.1-Flash` on NVIDIA A100/A800 using a SIF built
from the **official vLLM DeepSeek-V4.1 image** plus a pinned Ampere overlay.

## Base

```text
Official image : vllm/vllm-openai:deepseekv41-flash-0909
CUDA in image  : 13.0.1
GPU target     : A100/A800 (SM80)
SM80 overlay   : wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36
Output SIF     : deepseek-v41-flash-sm80-cu130.sif
```

For CUDA 12.9 use the [`cu129`](../../tree/cu129) branch.

## Why an overlay is still needed

The official DeepSeek-V4.1 image provides the model integration and native
CUDA/PyTorch/vLLM stack, but the stock CUDA path is not an A100 path: native
Triton FP8 E4M3 conversion requires SM89+, and the normal V4.1 sparse-MLA
selection targets newer GPU backends.

The pinned overlay adds the Ampere-specific pieces already exercised by the
A100 backport:

- SM8x `TRITON_MLA_SPARSE_DSV41` routing
- software E4M3 encode/decode for FP8 cache operations below SM89
- V4.1 128-token indexer pages
- CUDA-capable Triton sparse-MLA path reused from the ROCm/Triton backend
- pre-SM89 JIT warmup fixes
- DSpark V4.1 hooks from the validated backport pin

The base image's compiled native extensions are **not replaced**. See
[`patches/OVERLAY_MANIFEST.md`](patches/OVERLAY_MANIFEST.md).

## Build the SIF

```bash
git clone https://github.com/stozpark/deepseek-v41-flash-sm80.git
cd deepseek-v41-flash-sm80
git checkout main
./build_sif.sh
```

Result:

```text
deepseek-v41-flash-sm80-cu130.sif
```

`build_sif.sh` first prepares the exact SM80 overlay from the pinned commit,
then invokes Apptainer/Singularity using `Singularity.def`.

If you already have a checkout of the backport, including on a machine where
you want to avoid another clone:

```bash
BACKPORT_SOURCE=/path/to/vllm-backport ./build_sif.sh
```

For a fully pre-vendored build:

```bash
BACKPORT_SOURCE=/path/to/vllm-backport ./prepare_sm80_overlay.sh
SKIP_OVERLAY_PREPARE=1 ./build_sif.sh
```

The generated `patches/overlay/` and `*.sif` are intentionally git-ignored.
The SIF is several GB and should not be committed to Git.

If your installation cannot use fakeroot:

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

This `main` branch is CUDA 13.0.1. On an older data-center driver, prefer the
`cu129` branch unless your site has a validated CUDA forward-compat setup.

## Serve

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
./serve_tp8_a100.sh
```

Default serving configuration:

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

To isolate startup problems, disable DSpark first:

```bash
DISABLE_DSPARK=1 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

Enable expert parallel after the TP8 baseline is confirmed:

```bash
ENABLE_EXPERT_PARALLEL=1 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

## Verify inside the SIF

```bash
apptainer exec --nv deepseek-v41-flash-sm80-cu130.sif \
  python3 verify_runtime.py
```

This verifies the SM8x backend registration, the software FP8 helper, the
V4.1 Ampere attention module and the exposed GPU compute capabilities.

## Notes

This repository packages a reproducible A100 port; it does not claim that the
unmodified official Docker image itself supports A100. The official image is
the base, and the pinned SM80 Python/Triton overlay is what supplies the
Ampere execution path.
