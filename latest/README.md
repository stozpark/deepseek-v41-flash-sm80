# vLLM 0.32.0 / DeepSeek-V4.1-Flash-1017 / A100 SM80 migration

> **Port in progress — NOT A100-serving-validated.** This is an isolated workstream.
> The repository root's CUDA 13 image + 0909 25-file patch remains a historical,
> independently replay-verified baseline. **Do not apply that patch to vLLM 0.32.0.**

## Ground truth (2026-10-18)

- Model: `deepseek-ai/DeepSeek-V4.1-Flash-1017`. The official vLLM
  [recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) selects
  this new checkpoint; same config/tokenizer as 0909, different weights.
- Runtime: vLLM **v0.32.0**, commit
  `5df63a7203cdb9328d824a27b682e77c7b9928be`.
  Container: `vllm/vllm-openai:v0.32.0` (CUDA 13.0 default).
- A100 80 GB ×8 (TP8) is the first functional target. The checkpoint has
  **o_groups=8**, so native TP16 violates its grouped output projection
  topology. On 16 GPUs, run **two independent TP8 replicas** on one host
  (`serve_two_tp8.sh`) or one TP8 server per 8-GPU host. TP8×DP2 or
  TP8×PP2 require separate distributed integration and verification;
  PP cuts through index-sharing groups are unsupported upstream.
- Two TP8 replicas use separate copies of the Engram CPU-offload tables; provision substantial RAM (preferably 512 GiB+ for a dual-server diagnostic) and do not assume 16 GPU HBM eliminates host-memory constraints.\n- The official recipe does **not** claim A100 support. Upstream
  [issue #57144](https://github.com/vllm-project/vllm/issues/57144) tracks the
  missing SM80 sparse MLA, indexer, FP8 cache and other fallbacks.

## Migration warning

The current `main` overlay targets the launch-only
`deepseekv41-flash-0909` image. In vLLM v0.32.0 the model source package
is `vllm/models/deepseek_v41` rather than
`vllm/models/deepseek_v4_1`. The old patch also overrides code
that upstream has since optimized (new sparse indexer, Engram, mHC,
vision CUDA graphs, SWA bounded replay). Blind overlay or renaming is unsafe.

## Bring-up gates

1. **Baseline/source audit.** Use `latest/audit_source.py --mode baseline`
   in an unpatched 0.32.0 image; verify image, CUDA, expected v41 package.
   This is deliberately **not** a passing A100 deployment.
2. **Port SM80 kernels, then source audit.** Implement in the v0.32.0
   source tree; **do not replace entire upstream model modules**:
   - Ampere sparse MLA + SWA backend selection and engine registration.
   - Software/TileLang FP8 E4M3 cache write/read (including V4.1 compressed
     KV, indexer K, Engram) without SM89 native FP8 MMA.
   - DeepGEMM-free paged-MQA logits/indexer (128-token block pages),
     two-level candidate filtering, and large-context int64 addressing.
   - MXFP4 expert Marlin or bf16-dequant, MXFP8 linear fallback,
     no unconditional Hopper/Blackwell CuTe/DeepGEMM dispatch.
   - Correct WO_A and mHC fallback, preservation of vision and DSML parsers.
3. **Unit + kernel correctness on an A100.** At least tail mask, fragmented
   block tables, strided gather, >65K positions, chunked prefill,
   candidate mask, FP8 software conversion, Engram lookups.
4. **Actual TP8 model load.** Text-only 2K/8K first, then vision, tools,
   reasoning, 32K/128K; compare deterministic outputs or logits against
   a verified reference. Enable graphs, prefix cache, and speculative
   decoding **only after** an eager baseline passes.
5. **16 GPU.** First validate TP16 on a machine with 16 *locally visible*
   A100s. For 8+8 multi-node, produce separate distributed validation;
   `TP8×PP2` is not a drop-in substitute.

## Local commands (diagnostic, not proof of functionality)

```bash
# From inside the pinned upstream vLLM 0.32.0 container:
python3 latest/audit_source.py --mode baseline

# Static audit once a true SM80 port exists:
python3 latest/audit_source.py --mode port

# Only print server commands; never requires a GPU:
DRY_RUN=1 TP_SIZE=8  bash ./latest/serve_a100.sh
DRY_RUN=1 bash ./latest/serve_two_tp8.sh

# Actual launch requires a separately built and verified SM80-enabled SIF;
# the original 0909 SIF is intentionally rejected.
SIF_PATH=/path/to/v0320-sm80-verified.sif \
  MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 \
  TP_SIZE=8 ./latest/serve_a100.sh
```

## Tested vs untested

This track records an upstream version/architecture audit and a controlled
launcher. **It does not yet contain a validated SM80 kernel patch for 0.32.0,
a completed v0.32.0 SIF, or a physical A100 serving benchmark.**
Use the existing historical baseline only for comparison; not as evidence
that the new release runs on A100.

## References

- [vLLM V4.1 model recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash)
- [vLLM 0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)
- [A100 SM80 support issue](https://github.com/vllm-project/vllm/issues/57144)
- [Earlier working SM80 implementation](https://github.com/wtdcode/vllm-backport)
- [This repository's GLM-5.3 SM80 reference](https://github.com/stozpark/glm53-flash-sm80-cu129)

## Implemented migration artifacts

- `latest/VERSION.env`: v0.32.0, CUDA 13 default image, 1017 model.
- `latest/port_sm80_stage1.py`: exact-source guarded insertion of Ampere
  class/registry into modern `deepseek_v41` package; copies pinned FP8
  helpers and Triton paged-MQA fallback from the older backport.
- `latest/ampere_sparse.py`: provisional SM80 backend adapter based on ROCm
  reference Triton path. Registration alone is NOT sufficient for inference.
- `latest/Singularity.def`, `latest/build_sif.sh`: produce an explicitly
  marked **stage1** CUDA13 SIF with source/bytecode checks.
- `latest/audit_source.py`: baseline vs stage1 source audits; optional
  physical A100/CUDA checks and `o_groups` model verification.
- `latest/serve_a100.sh`: experimental TP8 launcher, text-only default,
  eager-first configuration. Requires ALLOW_EXPERIMENTAL=1.
- `latest/serve_two_tp8.sh`: two independent TP8 servers when 16 local
  GPUs are available; TP16 is rejected.

Build from repository root:
```bash
bash ./latest/build_sif.sh
# After source-level stage1 build, and only for a diagnostic A100 run:
ALLOW_EXPERIMENTAL=1 SIF_PATH="$PWD/deepseek-v41-flash-sm80-v0320-stage1-cu130.sif" \\
  MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 \\
  bash ./latest/serve_a100.sh
```

This does not imply that A100 model serving has passed. It is a reproducible
first integration step with explicit errors instead of a false success.
