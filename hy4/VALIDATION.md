# Phase 1 source candidate validation (2026-11-27)

Implementation commit: [570051c](https://github.com/stozpark/deepseek-v41-flash-sm80/commit/570051cc43ca58571356c94a540a1637632804bd).

| Gate | Status | Evidence |
|---|---|---|
| Source pin | PASS (source inspection) | Open vLLM PR #68633 head 3470b0db1820f699960d1376223602739a0f4ac3 fetched; flagship VL architecture present in registry and HYV4VLForConditionalGeneration declares SupportsPP. Not the official image digest. |
| Replacement anchors | PASS (source inspection) | Fetched three exact upstream blobs; all nine replacement anchors matched exactly once. Source manifest records blob IDs. |
| Branch integrity | PASS | Fetched branch HEAD and recursive tree after ref update; all seven files present. DeepSeek source left unchanged by implementation commit. |
| Sink math | PASS (CPU) | Exact committed helper and test copied to local CPU PyTorch 2.10.0 environment; git blob IDs independently matched GitHub: helper 2b0e40c3b3db79a3757933ba5739a9a73d37e005, test 928be579c4b7c26b547404ed73e90db1e0fab1df. pytest: 8 passed in 1.04s. Tests compare explicit zero-value sink, BF16/FP32, local head counts 4/8/16, all-masked rows, extreme logits, invalid inputs. |
| Complete pinned checkout replay | NOT RUN locally | Container could not resolve github.com; the connector can retrieve source blobs, not clone/build the upstream tree in this container. Run prepare_overlay.py --check, then apply and git diff --check on a clean checkout. |
| GPU kernels / compilation | NOT RUN | No CUDA GPU in execution environment. Four candidate GPU tests added (split-KV 1/4, rotary layouts NeoX/interleaved). |
| Full model / TP8 x PP2 / 128K | NOT RUN | Requires 16 physical A100-80GB and model weights. |
| CUDA 12.9 SIF | NOT BUILT | Pinned source uses PyTorch 2.14.0; resolve binary/toolchain support. |
| Performance/quality vs reference | NOT RUN | Do not infer TPS or equivalence from CPU helper tests. |

## Additional file-level observations

All upstream code links below refer to the exact source pin, not mutable main.

- [MLAAttention](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/layers/attention/mla_attention.py) accepts a model-provided backend class and instantiates its get_impl_cls(), supporting the model-specific subclass design. The generic backend still cannot serve learned sinks by itself.
- [Compressed-tensors dispatch](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py) chooses W8A16 on hardware below W8A8 minimum capability 89. [W8A16 scheme](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py) declares minimum capability 75 and block scale alias. No capability override needed for linear path.
- [Compressed-tensors MoE FP8 method](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_fp8.py) dispatches through the FP8 MoE oracle and forwards swiglu_limit. [Marlin experts](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py) declare support for static 128-block FP8 weights and >= SM75. Runtime oracle selection, pack scales, clamp behavior and actual TP shapes are still unverified.
- [PP transfer helper](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/models/utils.py) already passes Top-K indices if a stage splits a sharing group; the candidate fixes the model's wrong token-axis slicing. Need to test stage boundary with shared/full indexers and 1D/2D positions.
- Generic PR #68633 is still open and changes quickly; freeze the exact source commit. Do not assume later commits or an official image have the same API. The old CUDA 12.9 DeepSeek image cannot be reused as a Hy4 runtime merely by copying Python sources.

## Next acceptance gates

1. Clean pinned source replay, dependencies and native extension build. Verify CUDA driver/toolchain on target node before creating SIF.
2. Run candidate GPU kernels and upstream #68633 tests on A100; add actual paged MQA, native CUDA indexer K insertion, non-fused 1D FP8 query and stride/mask tests.
3. Verify compressed-tensors Marlin numerical equivalence for linear and MoE, full model fixture and multi-rank PP intermediate state. Do not lower capability gates, omit sink/gate, or use dense fallback.
4. Full checkpoint generation and matched-reference correctness before throughput tuning. Start short-context eager text-only; add CUDA graphs/prefix, actual 128K and vision separately. 1M and DFly2 remain independent research milestones.
