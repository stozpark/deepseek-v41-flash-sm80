> **Historical Phase 0 plan.** CUDA 12.9 assumptions are superseded by [the CUDA 13 port](cu13/README.md). Do not use the old branch for Hy4. This document preserves the original research evidence.

# Hy4-FP8 on A100 (SM80) — Phase 0 investigation

Status: research only; no Hy4 kernel patch, image build, GPU run, or speed/quality claim yet.
Survey date: 2026-11-26. Target: the public **tencent/Hy4-FP8** flagship (not Hy4 Flash or Hy4-preview).
Branch base: `cu129` at `f87c00011719bfa9995ccb549a0e599a5786a48a`. Existing DeepSeek scripts remain DeepSeek-only and must not be used to serve Hy4.

## Evidence and upstream map

| Source | Verified scope / reported evidence | Use for this port |
|---|---|---|
| [Tencent Hy4](https://github.com/Tencent-Hunyuan/Hy4), [checkpoint config](https://huggingface.co/tencent/Hy4-FP8/raw/main/config.json), [official vLLM recipe](https://recipes.vllm.ai/tencent/Hy4), [recipe PR #68844](https://github.com/vllm-project/vllm/pull/68844), [model PR #68296](https://github.com/vllm-project/vllm/pull/68296) | 78 layers, 770B backbone / ~774B stored, 49B active, block-FP8 W8A8 compressed-tensors (NOT preview MXFP8), 786 GB checkpoint; official image `vllm/vllm-openai:hy4` and 0.35.0 nightly. Recipe uses FlashMLA sparse, 128K context; DFly2 requires PP/DP/PCP/DCP=1. | Preserve model graph, exact checkpoint and parser. Resolve official image digest, embedded source commit, torch/CUDA/Triton versions and compatibility with target driver before overlay. |
| [Our GLM SM80 port](https://github.com/stozpark/glm53-flash-sm80-cu129/blob/main/README.md) | Owner reports actual A100 80GB x8 serving with TP8/EP, Marlin, BF16 main KV, Triton sparse MLA, indexer software FP8, MTP3. 512K configured does not establish full 512K or 1M request success. | Primary known-working local reference. Split-KV, int64 address safety, tail masks, 1D/2D lengths, pinned-buffer race fixes and offline SIF packaging. |
| [Our DeepSeek port](https://github.com/stozpark/deepseek-v41-flash-sm80) | Minimal replay-verified overlay, physical A100 inference not confirmed by owner. | Secondary source only: overlay generation, architecture dispatch and verification harness. Do not claim E2E validation. |
| [vLLM PR #47629](https://github.com/vllm-project/vllm/pull/47629) | Contributor reports 41 MQA and 53 sparse MLA kernel tests passing on 8x A800, GLM-5.2 NVFP4 E2E, 32.3 tok/s single stream, 100.3 tok/s aggregate at concurrency 4. PP and long-context tests are GLM, not Hy4. | Tested ancestor and regression cases; adapt API to Hy4 image. |
| [vLLM PR #68633](https://github.com/vllm-project/vllm/pull/68633) | Open Nov 19; portable SM80+ sparse MLA (512/576), software E4M3FN, Triton paged MQA, int64 offsets and next_n handling. CPU checks passed; author explicitly lacked physical SM80 testing. BF16/FP16 main KV, no DCP. | Strong recent generic backend candidate; compare against local tested GLM code, not blind cherry-pick. |
| [vLLM PR #69779](https://github.com/vllm-project/vllm/pull/69779) | Open Nov 25; Hy4 **Flash** on **SM89 Ada**, NOT flagship or SM80. Contributor reports 10 tests passing on L40 and 4x L40 official FP8 text-only streaming smoke, 44/50 GSM8K vs 45/50 eager control; no speed, full equivalence, multimodal or CUDA graph claim. Built on #68633. | Critical sink-aware adapter: get FP32 LSE from sparse kernel, rescale for learned zero-value sink; no sink-dropping dense prefill. Reuse design only, not Flash/SWA/IndexPool-specific code. |
| [HY V4 sink backend](https://github.com/vllm-project/vllm/blob/a9dcfae9/vllm/models/hy_v4/nvidia/flashmla_sparse.py), [sink bug #54568](https://github.com/vllm-project/vllm/pull/54568) | HYV4 uses a per-head learned attention sink. Dropping it changes attention; #54568 reports relative L2 0.2302 when omitted in an SM120 test. | Must implement exact sink semantics for prefill/decode, including split-KV and empty rows. No flag-only capability bypass. |
| [iHC fallback PR #55059](https://github.com/vllm-project/vllm/pull/55059) | Merged Sep 5; dispatch HPC-Ops -> Triton -> eager; H20/SM90 Hy4-preview tests, +24.74% output throughput vs eager. | Reuse native eager first, then validate Triton on SM80; do not port DeepSeek mHC blindly. |
| [PP CI PR #68604](https://github.com/vllm-project/vllm/pull/68604), [SM80 issue #66960](https://github.com/vllm-project/vllm/issues/66960) | Below-Hopper HYV4 skipped in upstream PP CI; no valid sparse backend. | Establish actual PP model/IndexCache boundary correctness; skip is not a fix. |
| [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport), [GPUStack DeepSeek A100 report](https://gpustack.ai/blog/self-hosting-deepseek-v4-1-flash-1017-on-8-nvidia-a100-gpus/) | Backport lists GLM-5.3 and DeepSeek-V4.1 support, not flagship Hy4 in inspected README; independent DeepSeek-V4.1-Flash-1017 on 8x A100 report uses latest-sm80. | Source of SM80 reference, not a confirmed Hy4 solution; pin exact commits. |
| [Hy4-preview dual H20 report](https://aik8s.run/ai-k8s/practices/hy4-preview-hardware-benchmark-plan/) | Reports 2x8 H20 SM90 BF16 with TP8 x PP2 (MTP off); preview, not released flagship. | Useful PP reference, not A100 proof. |
| [RonanLabs Hy4 on two Stations](https://github.com/RonanLabsAI/hy4-two-stations) | Independent full Hy4 FP8/GGUF measurements on two GB300 systems (not SM80). Reports FP8 vs Q4_K_M GSM8K 97.4% vs 97.0%, MMLU-Pro 83.8% vs 85.0%, Q4 top-1 agreement 91.5%, KL 0.146. Hardware placement differs; not a controlled GPU-speed comparison. | Real flagship comparison for GGUF reference, no claim that Q4 is universally equivalent or faster. |
| [FP8 KV RFC #59016](https://github.com/vllm-project/vllm/issues/59016) | Open; reports GLM-5.3 930K on A800; references #48374, #55173, #55184, #55177. | Later capacity optimization; must independently prove storage/dequant parity on Hy4. |

No public end-to-end run of the released flagship Hy4-FP8 on A100/A800 was found in the sources inspected. This is not proof none exists. The closest model-specific port is #69779 (Flash on Ada). No external PR above is evidence of our own GPU success.

## Model contract and non-negotiable correctness

- Checkpoint `HYV4VLForConditionalGeneration`, 78 layers, 64 query heads, 512 latent + 64 RoPE absorbed MLA, top-2048 DSA, indexer 32 x 128, learnable_sink=true, elementwise gated MLA, iHC four residual streams, M-RoPE.
- `indexer_types` starts full at layers 0,1,5,9,...,77 (0-based); other layers share preceding selection. Never invent indices at PP boundaries, re-compute with different weights, or leak indices across requests. Inspect original PP implementation and intermediate tensors first. If needed evaluate boundaries starting a full indexer (e.g. 37/41 or 41/37 layer counts for PP2), only after verifying loader and stage memory.
- Sink is a learned zero-value virtual entry in the attention denominator. With natural-log LSE from real selected keys, `out_sink = out_no_sink * sigmoid(lse - sink)`. Confirm LSE base, scaling, TP head order, fully masked rows, and correct split-KV merge before applying once; compare explicit zero-value sink reference. Do not set `supports_sink=True` without actually forwarding and consuming it. Dense prefill must be sink-aware or disabled.
- Preserve original q/k norm, Gated MLA, iHC and router semantics. Reuse upstream portable iHC/gated path where present. Disable HPC-only kernels via genuine capability dispatch.
- FP8 checkpoint is block-FP8 compressed-tensors, different from GLM/preview schemes: inspect per-tensor quantization, scales and MoE layer load/repack; prove compressed-tensors Marlin W8A16 on SM80 with BF16 activation against dequantized reference. Software FP8 for indexer cache is storage/codec, not native FP8 tensor-core computation.
- Use BF16 main MLA KV initially. No CPU fallback, attention densification, checkpoint requantization, skipped weight, disabled sink/gate or reduced layer count in final serving.
- DFly2 off at first (requires PP=1); checkpoint has no MTP layer. Prefix cache, CUDA graphs and vision initially off then independently enabled. Final multimodal support needs official preprocessing parity.

## Memory and topology (estimates, not measured)

Assume 80 GiB usable nameplate per GPU, decimal 786 GB checkpoint ~732 GiB. Uniform weight division over 16 GPUs is ~45.8 GiB/rank before stage imbalance, duplicated tensors, repack workspace, graph, communication and KV. GPU utilization=0.90 leaves only ~26 GiB/rank before those extras. Actual allocated weight memory must be measured after loading.

BF16 main MLA latent KV ~= (512+64)*2 = 1152 bytes/token/layer, before indexer keys, cache padding and other overhead. With replicated MLA KV under TP (verify actual implementation), 39 layers per stage yield 5.48 GiB at 128K, 10.97 GiB at 256K, 21.94 GiB at 512K and 43.875 GiB at 1M per GPU. Thus TP8 x PP2 cannot plausibly fit full 1M BF16 KV with ~45.8 GiB weights on 80 GiB; even 512K needs measurement. More PP stages (e.g. TP4 x PP4 within two nodes, verified boundaries) or storage-FP8 KV/DCP may help; none assumed supported.

| Candidate | Initial scope |
|---|---|
| TP8 x PP2 / EP off then on if supported | Preferred initial 16-GPU topology, 4K -> 32K -> 128K. Local TP limits cross-node collectives; PP partition and IndexCache semantics are a gate. |
| TP16 x PP1 | Fallback if PP fails, only with suitable interconnect and measured per-rank KV; cross-node collectives may be expensive. |
| TP4 x PP4 | Capacity exploration after correctness; stage balance and node mapping essential. |
| 1M context | Separate research milestone, not promised by startup max-model-len. Requires actual 1,048,576-token budget including output, measured cache and validated long sequence. |

## Work packages / gates

1. **Pin source & environment**: confirm official image digest/commit, checkpoint revision/hash and exact CUDA dependencies. Target R550/CUDA 12.9 compatible stack; official Hy4 image/nightly may be CUDA 13. Do not assume it runs on R550 or that old compiled extensions can be reused; if necessary build the new model-bearing source against supported CUDA 12.x, and separately validate native extensions. Preserve original DeepSeek branch behavior. Record source licenses/provenance/checksums.
2. **Audit dispatch & minimal overlay**: enumerate exact files in official `vllm/models/hy_v4/nvidia/`, shared attention, indexer, quantization, PP and vision code. Compare #68633 with locally tested GLM implementation. Backport only needed paths into a Hy4-specific overlay directory. Use CPU/static smoke and dummy-weight tests only as diagnostics, not proof of full inference.
3. **Kernel parity**: deterministic torch reference (FP32 accumulation) for 512+64 sparse attention, sink, Gated MLA, iHC, MQA indexer, top-k, software E4M3FN, and MoE quantization. Include empty rows, causal bounds, noncontiguous paged strides, 64-bit addresses, chunked prefill, equal-score tie policy, long contexts, TP-local sinks, and stage-boundary cases. Set numerical tolerances and top-k/logit criteria before testing; compare exact indices for non-ties, track near-ties separately.
4. **Single-node small fixture and multi-node startup**: check real SM80 dispatch, use a structurally faithful small fixture before downloading/loading full weights; then 16 physical A100s, Python frontend, eager, text-only, BF16 KV, no speculative decoding/prefix cache, short context and low concurrency. Inspect all ranks and per-stage weight/KV/peak memory; compare actual full checkpoint outputs/teacher-forced logits against a same-checkpoint reference on supported hardware where available. No claim of equivalence without reference.
5. **Runtime correctness**: deterministic prompts, repeated and interleaved requests, boundary tokens at every chunk/PP stage, tool parsing/streaming `hy_v4`, long-context needle, sampled tests, CUDA graphs/prefix-cache A/B, then vision preprocessing and image prompts. Compare known-working eager and optimized modes under matched conditions. Full 128K and higher requests explicitly measured. Stop and document mismatch rather than relax checks.
6. **Performance**: collect per-user decode TPS, aggregate TPS, prefill TPS, TTFT/TPOT, p50/p95, GPU utilization and network, peak VRAM, concurrent 1/4/8/16, 4K/32K/128K. Compare full Hy4 FP8 with same-prompt Q4_K_M baseline on A100 x8 (different hardware budgets disclosed) and if possible same hardware. Model-quality tests (math, coding, tool calls, long-context) with held-out benchmark, fixed templates and budgets. Later consider DFly2 separately only if PP/DFly constraints resolved; do not silently enable MTP.

## Status ledger

- Branch created: done.
- External survey and design: done (as-of date above).
- Exact Hy4 image/source pin and audit: pending.
- Hy4-specific kernel/overlay implementation: pending.
- Static, numerical GPU and full-checkpoint serving tests: pending.
- Performance / quality comparison: pending.

Do not mark the port as successful until physical A100 full-checkpoint generation, numerical/reference checks, and multi-node validation have passed.
