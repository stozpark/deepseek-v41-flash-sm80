# Review remediation (CUDA 13) — 2026-11-27

Baseline reviewed: 7495847. This change addresses all review categories with
code or explicit executable acceptance gates. It does not claim that the
GPU gates have run. No changes to the legacy DeepSeek build or main branch.

| Review item | Implementation / evidence |
|---|---|
| Explicit linear Marlin and actual dispatch | serve.sh and serve_fixture.sh set both backends; audited_worker verifies block Marlin Linear, MoE oracle + actual experts, packed tensors, sink alias, gate and indexer pattern on each rank, with eager prefill/decode trace. |
| Image packaging | image_verify.py generates/validates exact hashes of patched wheel files and port scripts, imports stable-ABI extensions and checks native op registration, imports actual Python modules and verifies CLI flags. Host imports do not prove physical CUDA ABI/launch success. |
| FP8 block scale and MoE | test_marlin.py tests repack and GEMM against CPU dequantized weights at TP8 local K/N, with Top-8 and active clamp; real checkpoint loaders/routing are not equivalent to this isolated test. |
| K-cache gather / Top-K | test_gpu_integration.py chains real native insert/gather, FP8 query, prefill and paged scores, actual upstream Top-K, physical KV conversion and sparse sink output with independent references. Near ties are bounded, not silently ignored. |
| >2/4GiB offsets | test_gpu_integration.py uses noncontiguous sparse pages whose byte offsets cross both boundaries, including native indexer write/gather and paged scoring. Requires ~4.1GiB temporary GPU VRAM. |
| Model-level execution | make_fixture.py, serve_fixture.sh, audited_worker and serve_probe exercise the real HYV4 model with synthetic dummy weights; fixture is not accuracy proof. |
| PP and NCCL | nccl_probe.py checks physical 16-rank TP and PP payloads; fixture PP1 vs PP2 includes a shared-index boundary. Full 37,41 vs 39,39 and official checkpoint reference remain separate runtime gates. |
| Long context and interleaving | serve_probe.py checks actual server token count, needle and chosen logprobs under matched runs/chunk settings; full 1M is excluded until cache capacity established. |

## Validation scope

CI runs the prior pinned upstream source replay, drift rejection, compilation
and CPU sink tests plus test_contract_cpu.py (manifest drift, selection oracle,
response parity, fixture immutability). No physical A100, Docker/SIF builder
or full model weights were available in the authoring environment. GPU and
model integration tests are candidate code that must be run on the real
cluster. No GPU success, full-checkpoint serving, latency or quality claim.

Follow-up source audit fixed the standalone persistent Top-K workspace
initialization and the VL fixture projector output dimension (upstream
HYV4VLConfig checks it even with vision disabled). Added GPU iHC/gate and
1D/2D trailing-64 M-RoPE math checks, top candidate logprob comparison,
and a rank/layer/phase coverage log validator with CPU negative tests.
The GPU tests remain unexecuted until a real SM80 runner is available.
