# Hy4 flagship SM80 experimental overlay

**Status: source candidate only, not installable production serving.** This work targets `tencent/Hy4-FP8`, not Flash or Preview. Source pinned to public open vLLM PR [#68633](https://github.com/vllm-project/vllm/pull/68633), head `3470b0db1820f699960d1376223602739a0f4ac3` (observed 2026-11-27). This is a development source tree, NOT a verified CUDA 12.9 binary or the official image. Never run the DeepSeek scripts for Hy4.

## Source changes and rationale

- Generic SM80 sparse MLA and MQA/FP8 codecs are supplied by the pinned PR, which must be evaluated against the real-A100 GLM port. The PR remains open. No claims of A100 runtime verification here.
- New Hy4-only adapter requests natural-log FP32 LSE from Triton for every sparse forward, applies the learned zero-value sink after split reduction, disables ShardQ via inherited capability, and refuses DCP. Existing Hy4 `_force_sparse_mqa()` prevents sinkless dense prefill. SM80 must fail rather than silently drop sinks.
- Patch Hy4's **trailing-64** fused indexer Q/RoPE FP8 write to use the generic PR's software E4M3FN codec below SM89. Preserve the native path, scale and head weight arithmetic. The generic DeepSeek fused indexer uses a different layout; cannot copy that whole kernel.
- Correct PP Top-K transfer to slice the token axis `full_num_tokens = positions.shape[-1]`, since M-RoPE positions can be [3,T]. Upstream already implements `pp_topk_buffers` and `copy_pp_buffers`; no new PP transport is needed until testing reveals a gap. Prefer starting the next stage on a full indexer anyway if stage memory allows.
- Compressed-tensors in this pin already routes FP8 W8A8 schemes below SM89 to `CompressedTensorsW8A16Fp8` (minimum capability 75), with block scale alias to `weight_scale_inv`. Do not lower W8A8 capability checks. Verify actual linear and MoE Marlin selection, weight packing, SwigLU limit, and shape alignment.
- iHC already has portable Triton/eager fallback. Do not replace it with DeepSeek mHC. Preserve original gate, weights, index reuse, and multimodal preprocessing.

## Reproduce the source overlay

```bash
git clone https://github.com/stozpark/deepseek-v41-flash-sm80.git
cd deepseek-v41-flash-sm80
git checkout hy4-sm80-cu129
git clone https://github.com/SiluPanda/vllm.git ../vllm-hy4-sm80
git -C ../vllm-hy4-sm80 checkout --detach 3470b0db1820f699960d1376223602739a0f4ac3
python hy4/prepare_overlay.py ../vllm-hy4-sm80 --check
python hy4/prepare_overlay.py ../vllm-hy4-sm80
git -C ../vllm-hy4-sm80 diff --check
```

The manifest pins the exact three upstream source blobs and replacement anchors. The script refuses dirty/drifted checkouts and syntax-checks all five files before writing. To regenerate, start from a fresh checkout; no fuzzy patching. Preserve upstream Apache-2.0 licensing. Source audit was done against the retrieved exact blobs; a clean checkout replay was not executed in this environment.

## Validation

```bash
# CPU only: exact zero-value sink reference
python -m pytest -q hy4/tests/test_sink_reference.py
# ONLY in the installed, patched vLLM environment on actual SM80:
python -m pytest -q hy4/tests/test_gpu_kernels.py
```

The GPU tests cover both split-KV choices, empty rows, and both rotary layouts with exact FP8-byte reference. Tests are a first gate, not an equivalence certificate. Next add real paged MQA / cache insert (including 1D positions), model/PP parity and compressed-tensors/Marlin numerical tests. Guard against upstream PR regressions.

## Build blocker / bring-up

The pinned source requires torch 2.14.0 (requirements/cuda.txt); no verified CUDA 12.9 wheel/image has been established for this revision. Official Hy4 nightly is CUDA 13-based in the recipe. Resolve exact CUDA/driver/extension compatibility with the A100 cluster first; rebuild against a compatible toolchain if required. Do not copy Python files from a different vLLM version into an old SIF. The branch name denotes the target environment, not a proven binary.

Only after GPU kernel tests pass: test structurally faithful fixture then full checkpoint on two 8x A100 nodes. First text-only eager, BF16 KV, `TRITON_MLA_SPARSE`, Marlin MoE, no DFly2, no prefix cache, no graph, 4K context/one sequence, TP8 x PP2 subject to verified layer split. Record per-rank memory, selected kernels, actual checkpoint generation and reference logits. Then 32K/128K, tool calling, CUDA graphs, vision, prefix-cache and performance. Do not advertise 1M until measured; TP8 x PP2 BF16 KV cannot fit 1M under the earlier estimates.

[Phase 0 plan](PORTING_PLAN.md) | [Pinned generic PR](https://github.com/vllm-project/vllm/pull/68633) | [Reference Hy4 Flash Ada adapter](https://github.com/vllm-project/vllm/pull/69779) | [Known-working local GLM port](https://github.com/stozpark/glm53-flash-sm80-cu129)
