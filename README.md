# DeepSeek-V4.1-Flash on A100/A800 (SM80) - CUDA 13

Run `deepseek-ai/DeepSeek-V4.1-Flash` with text + vision on NVIDIA A100/A800 using the official vLLM DeepSeek-V4.1 CUDA 13 image plus a verified production-minimal SM80 patch.

> For the current R550 A100 deployment, use the `cu129` branch. `main` requires a CUDA 13-capable driver/runtime stack.

## Target

```text
Model            : deepseek-ai/DeepSeek-V4.1-Flash
Official image   : vllm/vllm-openai:deepseekv41-flash-0909
CUDA in image    : 13.0.1
GPU target       : A100/A800 (SM80)
Default layout   : TP8 / PP1
Vision           : enabled by default (--mm-encoder-tp-mode data)
Speculative      : disabled for multimodal serving
Backport source  : wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36
Production delta : 25 files / 240149 bytes
Output SIF       : deepseek-v41-flash-sm80-cu130.sif
```

The production SIF applies `patches/generated/sm80-cu130-minimal.patch`; it does not copy the broad `patches/vendor/` tree. The broad snapshot is retained only for reproducible audit/regeneration. The CUDA 13 and CUDA 12.9 minimal deltas were independently generated and replay-verified against their respective official images and are byte-identical at the current pin.

The production patch preserves the official DeepSeek-V4.1 vision preprocessing/model wrapper and Engram DP/ubatching behavior while adding only the A100/SM80 runtime and selected post-0909 correctness fixes.

## Build

```bash
git clone https://github.com/stozpark/deepseek-v41-flash-sm80.git
cd deepseek-v41-flash-sm80
git checkout main
git pull origin main
./prepare_sm80_overlay.sh
./build_sif.sh
```

`build_sif.sh` verifies the exact CUDA 13 official image pin, the 25-file patch and SHA256, rejects non-A100 backend leakage, validates a supplied base SIF as CUDA 13.0, and validates the final SIF again.

If fakeroot is unavailable:

```bash
BUILD_ARGS="" ./build_sif.sh
```

## Serve: text + vision

Vision is ON by default:

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

Key defaults include:

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

The V4.1 vision wrapper does not support the MTP/DSpark draft-head weights, so the launcher rejects Vision + DSpark.

Normal multimodal serving:

```bash
ENABLE_VISION=1 DISABLE_DSPARK=1 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

DSpark may only be evaluated separately in intentional text-only mode:

```bash
ENABLE_VISION=0 DISABLE_DSPARK=0 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

## Physical A100 verification

```bash
singularity exec --nv deepseek-v41-flash-sm80-cu130.sif python3 verify_runtime.py
```

Then run long-context validation after serving:

```bash
python3 validate_long_context.py \
  --target-tokens 60000 \
  --deterministic-runs 4 \
  --sampled-runs 20 \
  --concurrency 4
```

CI verifies the exact production patch against the official image and verifies that the official multimodal preprocessing/wrapper remains intact. Physical A100 kernel/runtime validation is still required.
