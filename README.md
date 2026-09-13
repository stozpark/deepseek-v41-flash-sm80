# DeepSeek-V4.1-Flash on A100/A800 — CUDA 12.9 branch

This `cu129` branch builds `deepseek-ai/DeepSeek-V4.1-Flash` for NVIDIA
A100/A800 (SM80) using **CUDA 12.9.1** instead of the CUDA 13 image used by
`main`.

## Why CUDA 12.9

The pinned V4.1 Ampere port is `wtdcode/vllm-backport` commit:

```text
24cb31bb4fd0becee65c810c913a8caa4f610c36
```

The same source tree has an official CUDA 12.9 release build path using
`CUDA_VERSION=12.9.1` and the PyTorch CUDA 12.9 manylinux builder. CUDA 12.9 is
also the relevant boundary for several current vLLM CUDA/FP4 build paths.
Therefore this branch uses **12.9.1**, not 12.4.

The SM80 DeepSeek V4.1 runtime changes are the same as `main`:

- Triton sparse MLA for SM8x
- software E4M3 encode/decode on pre-SM89
- 128-token V4.1 indexer pages
- Engram CPU offload
- DSpark-5 with local argmax reduction

## Status

- Source/build configuration: pinned to a known V4.1 SM80 backport and its
  CUDA 12.9-supported build recipe.
- A100 V4.1 model/runtime path: publicly exercised with the same backport line.
- **This repository's CUDA-12.9 image itself has not been end-to-end executed on
  an A100 by this repository yet.** Run `verify_runtime.py` and the smoke test
  before treating it as production-qualified.

## Driver compatibility

NVIDIA documents CUDA 12.x minor-version compatibility for drivers **>= 525**
(and the 525–579 range is the CUDA-12 minor-compatibility range). Therefore an
R550 data-center driver is in the supported range for CUDA 12.x.

There are still NVIDIA caveats: features that require a newer driver and PTX
that a driver cannot JIT may need a driver upgrade. This build explicitly
compiles for **SM80** to minimize dependence on generic forward-PTX paths.

## Hardware baseline

```text
GPU:       A100/A800 80GB x8
Parallel:  TP8
Host RAM:  256 GiB minimum practical target; 384 GiB+ preferred
Model:     official deepseek-ai/DeepSeek-V4.1-Flash
CUDA:      12.9.1
```

Engram CPU offload is roughly 196 GB, so host RAM matters.

## 1. Check the machine

```bash
./preflight.sh
```

## 2. Build the CUDA 12.9 Docker image

```bash
./build_docker_cu129.sh
```

Default local image name:

```text
deepseek-v41-flash-sm80:cu129
```

The script checks out the exact backport commit and builds its upstream
`docker/Dockerfile` with:

```text
CUDA_VERSION=12.9.1
BUILD_BASE_IMAGE=pytorch/manylinux2_28-builder:cuda12.9-78e737ad29420ffc4800e677c51e2a852caf8359
TORCH_CUDA_ARCH_LIST=8.0
```

To reuse an already checked-out source tree:

```bash
SOURCE_DIR=/path/to/vllm-backport ./build_docker_cu129.sh
```

## 3-A. Run with Docker

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
./serve_docker_a100.sh
```

## 3-B. Convert the local image to Singularity/Apptainer

```bash
./build_sif.sh
```

Output:

```text
deepseek-v41-flash-sm80-cu129.sif
```

The conversion uses Apptainer/Singularity's `docker-daemon:` transport, so the
local Docker image must already exist and the user must be able to access the
Docker daemon.

Then serve:

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
SIF_PATH=./deepseek-v41-flash-sm80-cu129.sif \
./serve_tp8_a100.sh
```

## 4. Default serving configuration

```text
TP                         8
max_model_len              262144
max_num_seqs               16
max_num_batched_tokens     16384
gpu_memory_utilization     0.90
KV cache                    fp8_ds_mla
Engram                      CPU offload
Prefix cache                on
DSpark                      5 tokens
CUDA graph                  FULL_AND_PIECEWISE
Custom all-reduce           off
NCCL                        Ring / Simple
```

If startup/debugging fails, first remove speculative decoding:

```bash
DISABLE_DSPARK=1 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

After the baseline works, enable DSpark again.

## 5. Verify that the image is really CUDA 12.9 + SM80-capable

Docker:

```bash
docker run --rm --gpus all \
  -v "$PWD/verify_runtime.py:/tmp/verify_runtime.py:ro" \
  --entrypoint python3 \
  deepseek-v41-flash-sm80:cu129 /tmp/verify_runtime.py
```

SIF:

```bash
apptainer exec --nv \
  --bind "$PWD/verify_runtime.py:/tmp/verify_runtime.py:ro" \
  deepseek-v41-flash-sm80-cu129.sif \
  python3 /tmp/verify_runtime.py
```

The verifier checks:

- `torch.version.cuda` begins with `12.9`
- visible GPU capability is SM80 when a GPU is attached
- DeepSeek-V4.1 model code exists
- the Ampere sparse backend file exists

## 6. API smoke test

After the server starts:

```bash
./smoke_test.sh
```

## Why not CUDA 12.4?

This branch intentionally does not claim CUDA 12.4 support. The current vLLM
source has build components whose supported/optimized path is gated at CUDA
12.9, while its release tooling explicitly builds CUDA 12.9.1. Using the
project's existing 12.9 release path is materially safer than inventing an
unvalidated 12.4 patch set.

## Patch lineage

See [`patches/SM80_PATCHSET.md`](patches/SM80_PATCHSET.md). The CUDA version is
changed here; the DeepSeek-V4.1 Ampere model/runtime patch lineage remains the
same.
