# CUDA 13 code audit — 2026-11-27

Scope: all Hy4-specific Python, shell scripts, source pin, workflow, and documented build/launch path at e8934f7. Inherited DeepSeek CUDA 12.9 files are not the Hy4 runtime. Existing `hy4-sm80-cu129` kept for history; do not use it.

## Findings and fixes

| Finding | Status |
|---|---|
| Wrong CUDA branch originally selected; misleading root README and old Phase 0 plan | CUDA 13 entrypoint documented; upstream Dockerfile pin uses 13.2.1 with cu132 dependencies. Dedicated Hy4 VERSION.env; no changes to legacy DeepSeek runtime. |
| Overlay checked HEAD and anchor strings but not full working tree | Exact Git blob checks, clean checkout, byte-for-byte applied verification and refusal of extra/staged changes; build must pass verification. |
| Sink helper mixed with GPU imports, no independent CPU test | Separate pure PyTorch helper; checks TP-local shape, FP32 LSE/sink and device; exact CPU test 7 passed in 1.19s under torch 2.10.0+cpu. Expected blob hashes: helper 38789f23ec3096aebbe1bf59307177f67b9bec71 and CPU test d4898552ba75c7c8b7fa653f3fbaa850acc8cc37. GPU path now avoids unused total LSE and rejects unexpected CP/LSE consumers. |
| FP8 query test used GPU float8 conversion to construct decoded reference | Transfer raw bytes and decode on CPU instead. |
| Previous MQA tests wrote synthetic page bytes only | Added native K insertion test with page stride padding, negative slot sentinel, zeros and small inputs; compare full bytes, scales and untouched guards. Actual GPU execution pending. |
| Build did not verify packaged Python overlay/CUDA | Explicit upstream CUDA version and non-precompiled build, image packaging check, image ID/SIF hash. |
| Launcher only checked model architecture and CUDA major | Added per-GPU SM80/80GB/BF16 smoke, cu132 check, model contract, 128x128 FP8 group, PP partition, 245 shards check, GPU tests before load. Not a full checkpoint hash. |
| CI only ran compile/shell/source replay | Added exact replay verification, deliberate drift-rejection test and CPU sink reference. New CI result must be independently checked. |

## Source evidence (exact pinned #68633 head)
- [Dockerfile](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/docker/Dockerfile): CUDA 13.2.1, Python 3.14, cu132 builder, torch_cuda_arch_list includes SM80. Do not assume existing cu130 binary can take these modules.
- [Hy4 attention](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/models/hy_v4/nvidia/attention.py): original fused indexer writes native FP8; current SM80 port uses existing RoPE and software query quantization. Original sink fallback can drop bias, guarded by explicit SM80 backend. Full gated MLA unchanged.
- [Hy4 model](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/models/hy_v4/nvidia/model.py): PP shared top-k transfer exists; fixed M-RoPE token axis. PP 37/41 is default, 39/39 exercises actual cross-boundary transfer.
- [Generic sparse kernel](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/v1/attention/ops/triton_mla_sparse.py): returns natural-log FP32 LSE and zero on empty rows after split reduction; sink applied once.
- [Native K-cache code](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/csrc/cache_kernels.cu): contiguous K input, int64 cache stride and slots, 1e-4 amax floor, data bytes followed by scales per page. HYV4 unfused path concatenates K into contiguous tensor; native quant conversion and gather still require physical SM80 tests.
- [FP8 MoE oracle](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/layers/fused_moe/oracle/fp8.py) and [Marlin experts](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py): support static 128 block weights and W8A16 effective activation, explicit marlin selection; real packed-weight/SwigLU/TP parity not proven.
- [CLI](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/engine/arg_utils.py) and [headless serving](https://github.com/SiluPanda/vllm/blob/3470b0db1820f699960d1376223602739a0f4ac3/vllm/entrypoints/cli/serve.py): mp multi-node and headless worker arguments exist. NCCL routing and real PP not tested.

## Limits and remaining gates

The container has CPU-only PyTorch and no Docker, Triton, physical GPU, or direct GitHub DNS. Only the standalone CPU sink helper was actually run locally; the earlier branch CI had source replay PASS at e8934f7. New source replay/CI, Docker/SIF, A100 runtime and exact model loading remain pending until evidence is available. The upstream #68633 is open/unmerged; cannot treat source compatibility as numerical GPU validation. Next add native gather, true paged indexer/top-k, iHC/gated/FP8 Marlin and PP equivalence tests before complete flagship serving. No throughput, 128K/1M or multimodal claim. DFly2 intentionally disabled because PP2 is unsupported. CUDA 13.2 matching driver R595 recommended; minor compatibility with R580+ has PTX/JIT limits. If host supports only cu130, choose a coherent CUDA 13.0 source/torch/builder pin rather than mixing ABI versions.
