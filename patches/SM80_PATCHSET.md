# DeepSeek V4.1 A100/SM80 patch set

Production target: `DeepseekV41ForCausalLM`, A100/A800 SM80, TP8/PP1, text + vision.

The verified minimal CUDA 12.9 delta contains 25 files and covers these functional groups:

- V4.1 Ampere sparse-MLA backend and routing (`TRITON_MLA_SPARSE_DSV41`).
- software FP8 E4M3 encode/decode for pre-SM89 cache/Engram paths.
- V4.1 128-token sparse indexer pages and paged-MQA Triton fallback.
- long-context correctness: 64-bit physical block offsets and tail-store masking.
- strided block-table addressing fix for cache gather.
- CuTeDSL Hopper-only capability guard.
- CUTLASS FP8 SM89+ capability gate so SM80 falls back to Marlin.
- mHC DeepGEMM capability guard / TileLang fallback.
- `launch_pdl` propagation used by shared V4 kernels reached by V4.1.
- post-0909 DeepSeek V4.1 Responses API text block normalization.
- SWA image-width helper that behaves correctly for both multimodal and intentional language-model-only serving.
- preservation of the official V4.1 Engram DP/ubatching implementation while adding SM80 FP8 decode support.

Not included in the production patch:

- AMD model backends.
- XPU model backends.
- CPU model backends.
- PP>1 raw-token fix, because the production layout is PP1.
- SM120-specific fixes.
- H20/SM90-specific fixes.

Vision support is retained from the official DeepSeek-V4.1 image rather than replaced wholesale by the backport.
