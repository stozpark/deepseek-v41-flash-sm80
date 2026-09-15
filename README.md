# DeepSeek-V4.1-Flash on A100/A800 (SM80) - CUDA 13

Run `deepseek-ai/DeepSeek-V4.1-Flash` with text + vision on NVIDIA A100/A800 using the official vLLM DeepSeek-V4.1 CUDA 13 image plus the pinned SM80 backport.

> For hosts such as R550 that cannot run the CUDA 13 image, use the `cu129` branch. That is the recommended branch for the current A100 deployment.

## Target

```text
Model          : deepseek-ai/DeepSeek-V4.1-Flash
Official image : vllm/vllm-openai:deepseekv41-flash-0909
CUDA in image  : 13.0.1
GPU target     : A100/A800 (SM80)
Default layout : TP8 / PP1
Vision         : enabled by default (--mm-encoder-tp-mode data)
SM80 backport  : wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36
Output SIF     : deepseek-v41-flash-sm80-cu130.sif
```

## Build

```bash
git clone https://github.com/stozpark/deepseek-v41-flash-sm80.git
cd deepseek-v41-flash-sm80
git checkout main
git pull origin main
./build_sif.sh
```

`build_sif.sh` refuses to build if the branch drifts away from the official CUDA 13 image. A supplied `BASE_SIF` is checked before build and the completed SIF is checked again; `torch.version.cuda` must be 13.0.x.

If fakeroot is unavailable:

```bash
BUILD_ARGS="" ./build_sif.sh
```

## Serve: text + vision

Vision is ON by default and DSpark is OFF for first bring-up:

```bash
MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

The launcher uses:

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

After the base TP8 path is healthy, DSpark can be enabled explicitly:

```bash
DISABLE_DSPARK=0 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

For an intentional text-only deployment only:

```bash
ENABLE_VISION=0 MODEL_PATH=/models/DeepSeek-V4.1-Flash ./serve_tp8_a100.sh
```

## Physical A100 verification

```bash
singularity exec --nv deepseek-v41-flash-sm80-cu130.sif python3 verify_runtime.py
```

The verifier checks CUDA 13.0, `DeepseekV41ForCausalLM` registry inspection, SM80 sparse-MLA routing, software FP8, CUTLASS-to-Marlin fallback, paged-MQA tail masking, and V4.1 strided block-table addressing.

After serving, run long-context validation:

```bash
python3 validate_long_context.py \
  --target-tokens 60000 \
  --deterministic-runs 4 \
  --sampled-runs 20 \
  --concurrency 4
```

## Notes

The compiled native extensions from the official vLLM image are retained. The repository patches only Python/Triton/runtime paths needed for Ampere and selected post-0909 correctness fixes. The broad vendored snapshot under `patches/vendor/` is retained for reproducible regeneration and source auditing.
