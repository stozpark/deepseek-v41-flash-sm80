# DeepSeek-V4.1-Flash on A100/A800 (SM80) - CUDA 12.9

Run `deepseek-ai/DeepSeek-V4.1-Flash` with text + vision on NVIDIA A100/A800 using the official vLLM DeepSeek-V4.1 CUDA 12.9 image plus a verified production-minimal SM80 patch.

## Target

```text
Model            : deepseek-ai/DeepSeek-V4.1-Flash
Official image   : vllm/vllm-openai:deepseekv41-flash-0909-cu129
CUDA in image    : 12.9.1 family
GPU target       : A100/A800 (SM80)
Default layout   : TP8 / PP1
Vision           : enabled by default (--mm-encoder-tp-mode data)
Speculative      : disabled for multimodal serving
Backport source  : wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36
Production delta : 25 files / 240149 bytes
Output SIF       : deepseek-v41-flash-sm80-cu129.sif
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

`build_sif.sh` checks the exact official `-cu129` base pin, CUDA 12.9 family, SHA256 and 25-file manifest, absence of non-A100 model backends, and CUDA 12.9 again inside both supplied base SIFs and the final SIF.

If fakeroot is unavailable:

```bash
BUILD_ARGS="" ./build_sif.sh
```

For a disconnected build, see `OFFLINE.md`.

## Serve: text + vision

Vision is ON by default:

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

### DSpark and Vision

The DeepSeek-V4.1 vision wrapper does not support the MTP/DSpark draft-head weights. Therefore the launcher **rejects Vision + DSpark** instead of silently starting an unsupported configuration.

For normal multimodal serving, keep the defaults:

```bash
ENABLE_VISION=1 DISABLE_DSPARK=1 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

DSpark may only be tested separately in intentional text-only mode:

```bash
ENABLE_VISION=0 DISABLE_DSPARK=0 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
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

The physical verifier exercises the original model-registry failure path, SM80 sparse-MLA routing, software FP8, CUTLASS-to-Marlin capability gating, paged-MQA tail masking, and V4.1 strided block-table addressing. CI separately verifies that the official V4.1 vision preprocessing/wrapper stays intact and advertises encoder TP-data support.

## Branch roles

- `cu129`: production branch for CUDA 12.9 / current A100 deployment.
- `audit-sm80`: cu129 full-source audit/staging branch.
- `minimal-sm80`: cu129 dependency-minimization/audit branch.
- `main`: separate CUDA 13 branch; use only on a driver/runtime stack that supports it.

Source-level CI is not a substitute for the final physical A100 kernel/runtime test.
