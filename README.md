# DeepSeek-V4.1-Flash on A100/A800 (SM80) - CUDA 12.9

Run `deepseek-ai/DeepSeek-V4.1-Flash` with text + vision on NVIDIA A100/A800 using the official vLLM DeepSeek-V4.1 CUDA 12.9 image plus a verified production-minimal SM80 patch.

## Target

```text
Model          : deepseek-ai/DeepSeek-V4.1-Flash
Official image : vllm/vllm-openai:deepseekv41-flash-0909-cu129
CUDA in image  : 12.9.1 family
GPU target     : A100/A800 (SM80)
Default layout : TP8 / PP1
Vision         : enabled by default (--mm-encoder-tp-mode data)
Backport source: wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36
Production delta: 25 files / 240149 bytes
Output SIF     : deepseek-v41-flash-sm80-cu129.sif
```

The production SIF does **not** copy the broad vendored backport tree. It applies `patches/generated/sm80-cu129-minimal.patch`, which was generated from and replayed against the pristine official `-cu129` image. The broad `patches/vendor/` snapshot remains only for reproducible regeneration and audit.

The minimal patch intentionally leaves the official vision/VL preprocessing implementation untouched while adding the A100/SM80 runtime pieces and selected post-0909 correctness fixes. It also preserves official V4.1 Engram DP/ubatching code.

## Build

```bash
git clone https://github.com/stozpark/deepseek-v41-flash-sm80.git
cd deepseek-v41-flash-sm80
git checkout cu129
git pull origin cu129
./prepare_sm80_overlay.sh
./build_sif.sh
```

Result:

```text
deepseek-v41-flash-sm80-cu129.sif
```

`build_sif.sh` checks all of the following before/after build:

- exact official `deepseekv41-flash-0909-cu129` base pin
- `CUDA_FAMILY=12.9.x`
- verified SHA256 of the 25-file production patch
- absence of AMD/CPU/XPU model backends in that patch
- CUDA 12.9 runtime in a supplied `BASE_SIF`
- CUDA 12.9 runtime again in the completed SIF

If fakeroot is unavailable:

```bash
BUILD_ARGS="" ./build_sif.sh
```

For a disconnected build, see `OFFLINE.md`.

## Serve: text + vision

Vision is ON by default and DSpark is OFF for initial bring-up:

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

Key defaults:

```text
--tensor-parallel-size 8
--mm-encoder-tp-mode data
--kv-cache-dtype fp8_ds_mla
--engram-config {"cpu_offload":true}
--enable-prefix-caching
--tokenizer-mode deepseek_v41
--enable-auto-tool-choice
--tool-call-parser deepseek_v41
--reasoning-parser deepseek_v41
```

Enable DSpark only after the base path succeeds:

```bash
DISABLE_DSPARK=0 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

Use text-only mode only intentionally:

```bash
ENABLE_VISION=0 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

## Physical A100 verification

```bash
singularity exec --nv deepseek-v41-flash-sm80-cu129.sif python3 verify_runtime.py
```

Then start the server and run long-context validation:

```bash
python3 validate_long_context.py \
  --target-tokens 60000 \
  --deterministic-runs 4 \
  --sampled-runs 20 \
  --concurrency 4
```

The physical verifier exercises the original model-registry failure path, SM80 sparse-MLA routing, software FP8, CUTLASS-to-Marlin capability gating, paged-MQA tail masking, and V4.1 strided block-table addressing.

## Branch roles

- `cu129`: production branch for CUDA 12.9 / current A100 deployment.
- `audit-sm80`: cu129 full-source audit/staging branch.
- `minimal-sm80`: cu129 dependency-minimization/audit branch.
- `main`: separate CUDA 13 branch; use only on a driver/runtime stack that supports it.

Source-level CI is not a substitute for the final physical A100 kernel/runtime test.
