# DeepSeek-V4.1-Flash on A100/A800 (SM80) - CUDA 12.9

Run `deepseek-ai/DeepSeek-V4.1-Flash` with text + vision on NVIDIA A100/A800
using the official vLLM DeepSeek-V4.1 CUDA 12.9 image plus a pinned SM80
backport overlay.

## Target

```text
Model          : deepseek-ai/DeepSeek-V4.1-Flash
Official image : vllm/vllm-openai:deepseekv41-flash-0909-cu129
CUDA in image  : 12.9.1 family
GPU target     : A100/A800 (SM80)
Default layout : TP8 / PP1
Vision         : enabled by default (--mm-encoder-tp-mode data)
SM80 overlay   : wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36
Output SIF     : deepseek-v41-flash-sm80-cu129.sif
```

The branch contains an official-image compatibility audit and a reproducible
compact delta under `patches/generated/`. The full vendored source snapshot is
kept under `patches/vendor/` so disconnected build hosts do not need GitHub.

## Build

```bash
git clone https://github.com/stozpark/deepseek-v41-flash-sm80.git
cd deepseek-v41-flash-sm80
git checkout cu129
git pull origin cu129
./build_sif.sh
```

Result:

```text
deepseek-v41-flash-sm80-cu129.sif
```

`build_sif.sh` refuses to build if `VERSION.env` / `Singularity.def` drift away
from the official `-cu129` image. If `BASE_SIF` or `BASE_URI` is supplied, its
PyTorch CUDA runtime is checked and a non-12.9 base is rejected. The completed
SIF is checked again before the script reports success.

If fakeroot is unavailable:

```bash
BUILD_ARGS="" ./build_sif.sh
```

For a fully offline build, see `OFFLINE.md`.

## Preflight

```bash
./preflight.sh
```

Recommended starting point:

```text
GPU      : 8 x A100 80GB
Host RAM : 384 GiB+ preferred
TP       : 8
PP       : 1
Context  : 256K first
```

## Serve: text + vision

Vision is ON by default.

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash \
./serve_tp8_a100.sh
```

The launcher adds:

```text
--tensor-parallel-size 8
--mm-encoder-tp-mode data
--tokenizer-mode deepseek_v41
--enable-auto-tool-choice
--tool-call-parser deepseek_v41
--reasoning-parser deepseek_v41
```

Bring-up defaults keep DSpark disabled until the base A100 path is confirmed.
After successful baseline serving:

```bash
DISABLE_DSPARK=0 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

For an intentional text-only deployment only:

```bash
ENABLE_VISION=0 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

## Verify on the physical A100

```bash
singularity exec --nv deepseek-v41-flash-sm80-cu129.sif \
  python3 verify_runtime.py
```

This checks the CUDA 12.9 runtime, DeepSeek-V4.1 model registry, SM80 sparse-MLA
routing, software FP8 path, CUTLASS-to-Marlin capability gate, paged-indexer
Triton fallback, and strided block-table gather behavior.

## Offline source integrity

```bash
cd patches/vendor
sha256sum -c SHA256SUMS
```

The base image's compiled native extensions are retained; the repository only
adds/patches the Python/Triton/runtime pieces required for the SM80 execution
path and selected post-0909 correctness fixes.
