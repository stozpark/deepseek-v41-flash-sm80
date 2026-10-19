# Upstream delta review: DeepSeek-V4.1-Flash-1017 SM80 (2026-10-19 KST)

## Baseline and search window

Pinned release: **vLLM v0.32.0**, commit `5df63a7203cdb9328d824a27b682e77c7b9928be`, released 2026-10-18 06:50 UTC. Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Review cutoff: 2026-10-18 15:33 UTC (00:33 KST Oct 19). Fetched upstream GitHub commits dated after release and recently updated issues/PRs. A release-tag comparison against `main` is **diverged**, so cherry-pick only audited, relevant fixes; DO NOT merge main into this port.

Key principle: the in-repository A100 branch uses TP8 / PP1, CPU-host Engram offload, FP8 indexer (not MXFP4 indexer), **no** speculative decoding, no DCP/PCP, no KV connector, eager attention. Hardware numerical tests are required; source-only CI is not evidence of physical A100 correctness.

## Release-to-main commits

At cutoff, 14 commits landed after the 06:50 UTC release. Checked relevant changes:

| Upstream commit | Subject | SM80 decision |
| --- | --- | --- |
| [072c0886](https://github.com/vllm-project/vllm/commit/072c0886d53d933a7e1d3da7e8a3a91670cf181d) | Retain plain CUDA arch if suffixed variant requested | Relevant **only** if compiling from source with SM80 target; official prebuilt v0.32.0 image + Python overlay does not need the build-system hunk. Do not blindly cherry-pick. |
| [b510857d](https://github.com/vllm-project/vllm/commit/b510857d37640209a4a3e01a13260ac26f2a93f3) | Build fused GDN MTP on SM80–SM90 with CUDA <13 | Not applicable: this recipe requires CUDA >=13 and disables speculative draft. |
| [e80b6c64](https://github.com/vllm-project/vllm/commit/e80b6c642f21fd5b24439825592f5b9ced36add8) | ROCm DSv4 shared expert/AITER overlap | Not relevant to A100 CUDA numerics. |
| [7c9042c0](https://github.com/vllm-project/vllm/commit/7c9042c05addf034366b9a1da5b1d3e3687c052b) | MRV2 speculative PP deadlock | Not applicable to TP8 PP1 and no speculation. |
| [83c9a547](https://github.com/vllm-project/vllm/commit/83c9a547bf9a3b4f2653fb5766886bdc5dba35be) | Include recipes in release Docker context | Already consuming the model recipe directly for configuration, not source vendoring. |

Other changes included docs, an SM103-only DeepEP image change, Qwen models, and OpenCV video. No release-after commit in this cutoff resolves the missing Ampere sparse-MLA/indexer.

## Outstanding unmerged PRs and issues

| PR or issue | Potential failure | Decision / regression gate |
| --- | --- | --- |
| [#62197](https://github.com/vllm-project/vllm/pull/62197) | **Accuracy:** inverse RoPE incorrectly reported as fused on fallback sparse decode. Reproduced upstream on gfx1151 including V4.1-1017 (GSM8K 1/25->25/25), not A100. | **PORTED:** exact-source patch in `port_sm80_stage1.py`. CPU branch contract `test_inverse_rope_contract.py` and real A100 output oracle `verify_sm80_numeric.py`. |
| [#61938](https://github.com/vllm-project/vllm/pull/61938) | **Memory correctness:** int32 overflow of split-K Triton MLA workspace offsets and unprofiled large workspace, on MI355 generic MLA backend | Different kernels from SM80 sparse path; **do not import whole patch**. Regression strategy: paged/block table address casts to int64 in SM80 Triton MQA and inspect all large `stride` products, long context 32K->128K. |
| [#60493](https://github.com/vllm-project/vllm/pull/60493) | Sparse indexer expanded block table allocated after memory profiling, OOM when sizing KV cache | Applicable memory-planning concept; upstream patch invasive and still open, so initial launch stays at 32K / 2048 batched tokens / 4 sequences, GPU utilization 0.90. Do not claim 1M context. |
| [#62208](https://github.com/vllm-project/vllm/pull/62208) | PP stage-entry Engram omitted -> wrong results | **Avoid:** PP1. Must port and test before any `--pipeline-parallel-size > 1`. |
| [#62269](https://github.com/vllm-project/vllm/pull/62269) | Engram DP-sharded CPU tables with CuMem allocator hang on NCCL all-to-all | Avoid `--enable-cumem-allocator` + Engram DP sharding combination. TP8 PP1 uses no Engram DP exchange. |
| [#62283](https://github.com/vllm-project/vllm/issues/62283) | KV offload across TP layouts without canonical layout can silently corrupt cache | No KV offload connector in initial config. Never attach shared noncanonical KV storage to these replicas. |
| [#62259](https://github.com/vllm-project/vllm/pull/62259), [#62290](https://github.com/vllm-project/vllm/pull/62290) | Truncated/unclosed DeepSeek DSML tool call may leak invalid partial JSON/markup | Unmerged Python/Rust frontend patches: **do not hotpatch compiled Rust**. Add request-level regression for truncated tool calls before production tool calling. For now treat tool behavior as not verified. |
| [#62226](https://github.com/vllm-project/vllm/pull/62226), [#62220](https://github.com/vllm-project/vllm/issues/62220) | `reasoning_effort=medium/minimal` rejected by old V4.1 tokenizer | No runtime patch: use supported `low`/`high` until upstream lands and passes both frontend tests. |
| [#60762](https://github.com/vllm-project/vllm/pull/60762), [#57028](https://github.com/vllm-project/vllm/pull/57028) | SM120 block sizes and FlashInfer sparse MLA serving bugs | Hardware-specific SM120 path, NOT the A100 SM80 fallback. |
| [#62240](https://github.com/vllm-project/vllm/pull/62240), [#62170](https://github.com/vllm-project/vllm/pull/62170) | BF16 DeepSelect / SM100 indexer optimization | DeepSelect SM100-specific; use SM80 **software E4M3 + BF16 Triton MQA** and `sparse_indexer_topk_backend=torch` as accuracy baseline. |

## A100 numerical gates

**Implemented** (source checks, not yet run on an actual A100):
1. E4M3FN software encode **bitwise** vs PyTorch across >23,000 values including tie, normal/subnormal boundaries, clipping, zero; exhaustive finite-byte decode; `verify_sm80_numeric.py`.
2. Independent PyTorch reference for compressed FP8 KV dequant -> BF16 sparse attention, sink score, SWA-only, ragged topk, reordered/duplicated pages, and inverse RoPE final output; `verify_sm80_numeric.py`.
3. FP8 indexer Q and paged K software fallback, FP8 prefill scores, paged decode with nonmonotonic block tables, causal/pad tails, candidate block selection and masked Top-K vs independent PyTorch oracle; `verify_sm80_indexer.py`.
4. Engram SM80 lookup byte-exact E4M3FN/E8M0, both packed table result and BF16 dequant, with out-of-vocabulary zero masking; `verify_sm80_engram.py`.
5. Explicit SM80 fail-closed gates for unsupported MXFP4 indexer, DCP/PCP, multi-token speculative decode. `--enforce-eager`, no SWA bounded replay/prefix cache.
6. Pin exact upstream source Git blob IDs before each patch; compileall + independent source replay CI; no broad overwriting of upstream `deepseek_v41`.

**Still mandatory** before service-ready designation:
- Build exact v0.32.0 CUDA13 SIF and execute all CUDA numeric tests on A100 SM80.
- Run weights-loaded TP8 on *1017*; compare logits/generation to independent supported reference at fixed seeds for text, vision, reasoning, tool calls; check 2K,8K,32K,128K and failure modes.
- Verify all cache/indexer writers and readers use identical 584-byte MLA and 132-byte FP8 indexer records, and **Engram pinned-host UVA path** matches the small GPU-table oracle; NVIDIA reader/writer kernels may need further patches even if isolated math is correct.
- At least 100 representative requests under multi-request prefill/decode with KV fragmentation and candidate block heterogeneity, 0 invalid outputs, no silent truncation; benchmark throughput only afterward.

**No A100 access is available in this conversation's tool runtime. No claim of a passed GPU numerical test or usable TP8 model checkpoint has been made.**
