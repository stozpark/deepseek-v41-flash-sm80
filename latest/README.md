# DeepSeek V4.1 Flash 1017 — vLLM 0.32.0 on A100 SM80

> **Experimental source integration only.** GitHub Actions validates exact
> upstream-source patch replay and Python/Bash syntax. **Real NVIDIA A100
> numerical parity, SIF build, and full checkpoint serving have NOT been
> verified yet.** Do not merge this branch or put its server on production.

## Immutable upstream baseline

- **Model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017`, verified against
  [the official vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash).
- **vLLM:** `v0.32.0`, upstream commit
  `5df63a7203cdb9328d824a27b682e77c7b9928be`,
  original container `vllm/vllm-openai:v0.32.0`.
- **CUDA:** 13.0 or newer. **GPU:** NVIDIA A100/A800 SM80, 8x80 GB, TP8.
- **16 GPU:** independent 2×TP8 replicas, not TP16 (`o_groups=8`). With
  two 8-GPU nodes, launch each host independently. Reserve sufficient system
  RAM for separate Engram host-offload tables; 512 GiB or more may be
  appropriate for two full replicas, depending on memory placement.
- Older root SIF, vendor snapshot, and `sm80-cu130-minimal.patch` target
  the **0909** branch and old `deepseek_v4_1` Python package. **Never
  overlay them wholesale onto v0.32.0's `deepseek_v41` implementation.**

## Work implemented

| Stage | Patch | Functionality | Verification in this repo |
| --- | --- | --- | --- |
| 1 | `port_sm80_stage1.py` | Exact-source guarded Ampere sparse MLA backend, SM80 helper module, CUTLASS FP8 guard, and upstream inverse-RoPE bugfix [#62197](https://github.com/vllm-project/vllm/pull/62197) | CI source replay and six-case `test_inverse_rope_contract.py` |
| 2 | `port_sm80_numeric.py` | Software E4M3FN encode/decode replacing SM80-incompatible native Triton casts in paged KV, compressor, indexer K, and sparse attention; preserves UE8M0 and FP4-rounded scale semantics | CI bytecode and exact-source checks, `verify_sm80_numeric.py` prepared for GPU |
| 3 | `port_sm80_indexer.py` | Remove DeepGEMM requirement on SM80, quantize Q to correct FP8 bytes, use BF16 Tensor Core Triton `fp8_mqa_logits_triton`/`fp8_paged_mqa_logits_triton`; preserve upstream two-level candidate masking | CI patch replay + `verify_sm80_indexer.py` prepared for GPU |
| 4 | `port_sm80_engram.py` | Avoid unsupported direct E4M3 tensor loads and FP32 casts in host-offloaded Engram tables on SM80; re-interpret bytes and preserve E8M0 scale semantics | CI patch replay + `verify_sm80_engram.py` tests both raw packed bytes and BF16 output |
| 5 | `port_sm80_dspark.py` | DSpark5 target verification: six speculative rows with distinct causal ends in SM80 paged MQA | CI replay, `verify_sm80_dspark.py` available for GPU |

**Accuracy policy:** source CI passing is **not** a successful numerical
benchmark. The real A100 script checks E4M3FN bitwise parity vs PyTorch,
all 256 E4M3FN codewords, mixed sparse MLA and SWA output vs independent
PyTorch attention, inverse RoPE and NoPE invariance, paged FP8 indexer
logits and rank-consistent Top-K, candidate-block masking, and causal tails. Engram lookup is checked against a separate byte-exact packed E4M3/E8M0 and BF16 reference.
Any assertion or unsupported configuration stops the launcher before model
load. `ALLOWED_EXPERIMENTAL` is NOT a bypass around numerical tests;
the required opt-in is `ALLOW_EXPERIMENTAL=1`.

## Source validation (no GPU required)

```bash
git fetch origin vllm-v0320-sm80
git switch vllm-v0320-sm80

# GitHub Actions automatically checks pinned vLLM-0.32.0 source, all patches,
# Python/Bash syntax, inverse-RoPE semantics and fail-closed reapplication.
# Validate the launch flags without a GPU:
DRY_RUN=1 bash latest/serve_a100.sh
DRY_RUN=1 bash latest/serve_two_tp8.sh
```

## Apptainer/Singularity build (CUDA 13)

```bash
# If connected to Docker Hub or with the pinned v0.32.0 image cached:
bash latest/build_sif.sh

# In the restricted corporate network, you can provide an existing SIF
# containing the exact upstream vLLM 0.32.0 + CUDA >=13:
BASE_SIF=/path/to/upstream-vllm0320-cu130.sif bash latest/build_sif.sh
```

The resulting default file is
`deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif`. The build
checks source patch integrity and inverse-RoPE completion, not CUDA runtime.

## **Actual A100 validation** (mandatory)

```bash
# This command first runs all diagnostics on GPU 0 using Apptainer --nv,
# then, only if all pass, attempts the 1017 TP8 server with safe constraints:
ALLOW_EXPERIMENTAL=1 \
MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 \
bash latest/serve_a100.sh
```

The launcher enables Vision by default (`ENABLE_VISION=1`) and offers opt-in DSpark5 (`ENABLE_DSPARK=1`). See [VISION_DSPARK_BRINGUP.md](VISION_DSPARK_BRINGUP.md) for tests and limitations. The launcher intentionally chooses 32K maximum context, 2048 batched
tokens, concurrency 4, eager attention, 128-token indexer pages, explicit
FP8 indexer, `sparse_indexer_topk_backend=torch`, optional DSpark5 (requires separate A100 multi-token test),
no DCP/PCP, no prefix caching or bounded SWA replay, no CuMem allocator.
Vision input still needs a real A100 semantic evaluation; setting its flag only enables the upstream model path.
The configured OpenAI API port is 18005.

Standalone GPU diagnostics (after SIF build):

```bash
apptainer exec --nv deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif \
  python3 /opt/deepseek-sm80/verify_sm80_kernels.py
apptainer exec --nv deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif \
  python3 /opt/deepseek-sm80/verify_sm80_numeric.py
apptainer exec --nv deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif \
  python3 /opt/deepseek-sm80/verify_sm80_indexer.py
apptainer exec --nv deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif \
  python3 /opt/deepseek-sm80/verify_sm80_engram.py
apptainer exec --nv deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif \
  python3 /opt/deepseek-sm80/verify_sm80_dspark.py
```

**Stop rules:** no inference claim if source or GPU numerical probes fail.
Before ready-to-merge require 1017 real weights TP8 load and deterministic
text + vision + tool + reasoning comparison with a supported reference,
multi-request tests with fragmented KV pages, indexer candidate routing
and context-length expansion (2K → 8K → 32K → 128K). CUDA graphs, prefix cache, pipeline/context parallelism and 1M contexts are unsupported
until separately verified. No model-quality results can be inferred
from bytecode tests alone.

## Upstream changes after v0.32.0

See [UPSTREAM_AUDIT_2026-10-19.md](UPSTREAM_AUDIT_2026-10-19.md).
Critical numerical fix [#62197](https://github.com/vllm-project/vllm/pull/62197)
is integrated. Evaluated but not blindly imported:
[#61938](https://github.com/vllm-project/vllm/pull/61938) (Triton
MLA int32 overflow),
[#60493](https://github.com/vllm-project/vllm/pull/60493) (indexer metadata
memory sizing),
[#62208](https://github.com/vllm-project/vllm/pull/62208) (PP-stage
Engram correctness), and
[#62259](https://github.com/vllm-project/vllm/pull/62259) /
[#62290](https://github.com/vllm-project/vllm/pull/62290) (truncated DSML
tool-call output).

## Runtime compatibility fix: missing `VLLM_INDEXER_LOGITS_MAXNREG`

An A100-SXM4-80GB / CUDA 13.0 / PyTorch 2.13.0 diagnostic reached
`SM80_SOFTWARE_FP8_LUT_PASS` then failed importing
`vllm.v1.attention.ops.mqa_logits_triton` with
`AttributeError: vllm.envs has no VLLM_INDEXER_LOGITS_MAXNREG`.

The vendored SM80 MQA helper was inherited from an earlier fork. The exact
vLLM 0.32.0 base does **not** register three indexer tuning names; the
updated vendored helper reads them from `os.environ` directly, preserving
historical defaults: `MAXNREG=0` (unconstrained),
`KV_GROUP_MIN_M=0` (auto), and `FACTOR_K_SCALE=1` (enabled).
An independent CPU test `test_mqa_env_compat.py` now guards all three
defaults/overrides against the pinned upstream `vllm/envs.py` during source
CI and SIF build. This fixes the import API mismatch, **not** proof of
correct A100 numerical execution.

The default SIF name was changed to `stage5-envfix` to prevent accidentally
rerunning the old, already-built `stage5-unverified` image.

```bash
git switch vllm-v0320-sm80 && git pull --ff-only
bash latest/build_sif.sh
SIF=./deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif

# Check import in the NEW image, with GPU:
apptainer exec --nv "$SIF" python3 -c \
 'from vllm.v1.attention.ops.mqa_logits_triton import fp8_paged_mqa_logits_triton; print("SM80_MQA_IMPORT_PASS")'

# Resume safe hardware test sequence (still experimental):
SIF_PATH="$PWD/deepseek-v41-flash-sm80-v0320-stage5-envfix-cu130.sif" \
 ALLOW_EXPERIMENTAL=1 MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 \
 bash latest/serve_a100.sh
```

The A100 SM80 adapter imports `vllm.models.deepseek_v41.amd.rocm` for
the portable attention implementation. This eagerly imports
`vllm.platforms.rocm`, whose top-level optional AMD probes print
`No module named amdsmi`, `No module named vllm._C`, and
`No module named vllm._rocm_C`. On an NVIDIA-only CUDA 13 image these
three warnings are expected from this inherited ROCm path; installing AMD
tools or an old vLLM `_C` module is **not** the correct response.

**Different extension:** vLLM v0.32.0's actual CUDA platform requires
`vllm._C_stable_libtorch`. The A100 probe explicitly asserts both
`current_platform.is_cuda()` and import of `_C_stable_libtorch`, emitting
`VLLM_CUDA_EXTENSION_PASS`. If that assertion fails, treat it as a genuine
CUDA image/version problem. If it passes and only the three ROCm warnings
remain, look for the first later `Traceback` / failed numerical assertion
rather than installing `amdsmi` or hiding warnings.
