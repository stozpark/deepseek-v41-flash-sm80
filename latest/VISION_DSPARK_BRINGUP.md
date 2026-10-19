# A100 SM80 Vision + DSpark5 bring-up (vLLM v0.32.0)

Status: **source integration; real A100 accuracy and full checkpoint unverified**.
This adds functionality to the existing four correctness stages without
replacing them or the pinned vLLM source. This does not copy outdated PP8
specific patches.

## Community approaches compared

- `wtdcode/vllm-backport`: TP8/SM80 V4.1 DSpark5, CUDA graphs, Marlin,
  software FP8. It is an earlier fork, not a drop-in for vLLM 0.32.0.
- `Zanooda/deepseek-v4.1-flash-cmp170hx`: Vision and DSpark on SM80 using PP8.
  They kept draft model on the last rank and replicated PP-crossing KV/indexer
  state; TP8/PP1 does **not** need this particular topology workaround.
- `Tokha233/deepseek-v4.1-flash-a100-turbo`: fast A800 TP4xDP2 DSpark,
  but their documented launch passes `--language-model-only`, **not Vision**.

## Vision (no PP8 dependency)

The upstream `deepseek_v41/nvidia/vl_model.py` already has the ViT/aligner
and a `get_mtp_target_hidden_states` hook. Launch with `ENABLE_VISION=1`
and data-parallel vision encoder. This is now the default unless explicitly
disabled (`ENABLE_VISION=0`). Correct handling requires checking
image-span sentinel input IDs, image dimensions/position ordering,
image MoE route, multi-image and multi-turn image followups. A live
image-serving API smoke exists at `smoke_vision_dspark.py`, but API smoke
**alone does not prove visual semantic accuracy**.

## DSpark5 (A100 TP8)

The upstream v0.32.0 DSpark model and draft loader already exist. Prior
SM80 patch explicitly refused `next_n != 1`; `port_sm80_dspark.py`
now permits **1..6 target rows per request** (one verified token plus
five speculative), keeping FP8 + no PCP/DCP requirements.

**Critical correctness change:** original vendored paged MQA used the
last sequence end for *all* speculative rows. Updated
`fp8_paged_mqa_logits_triton` accepts `[B,next_n]` effective
sequence lengths and applies the causal mask per query row. The independent
A100 test `verify_sm80_dspark.py` compares different row ends, nonmonotonic
physical page tables, and complete masked tails against PyTorch.

`ENABLE_DSPARK=1` selects `method=dspark` and `num_speculative_tokens=5`.
DSpark remains opt-in until this kernel test and the full-model drafting/
verification comparison pass. Disabled by default.

## Required on real A100

```bash
bash latest/build_sif.sh

# Config dry-runs do not require GPU:
DRY_RUN=1 ENABLE_VISION=1 ENABLE_DSPARK=0 bash latest/serve_a100.sh
DRY_RUN=1 ENABLE_VISION=1 ENABLE_DSPARK=1 bash latest/serve_a100.sh

# ALL four baseline numerical guards still run before server start.
# DSpark adds a fifth real GPU check:
ENABLE_VISION=1 ENABLE_DSPARK=1 ALLOW_EXPERIMENTAL=1 \
 MODEL_PATH=/models/DeepSeek-V4.1-Flash-1017 bash latest/serve_a100.sh

# When server health is ready, execute independent application checks:
python3 latest/smoke_vision_dspark.py --vision --dspark
```

### Hard-blocked acceptance criteria

1. Zero errors in real A100 baseline kernel tests plus DSpark5 multi-query
   paged cache causal-mask and Top-K comparison.
2. Text-only and Vision-only full checkpoint load first (same 1017 weights).
3. With DSpark enabled, compare *target* acceptance and output distribution
   against a verified speculative-off reference at fixed random seeds.
   Sampling output strings need not match bitwise in stochastic modes;
   ensure token distribution and target verification are correct.
4. Vision+DSpark: actual image prompt produces meaningful image-dependent
   answer, not a text-only hallucination; check multiple images and ordering.
5. Throughput/latency and DSpark acceptance measured separately:
   eager without DSpark, eager with DSpark, then CUDA graphs if stable.
   No speedup claims without profiling.
6. Document memory usage and failure cases at 2K/8K/32K/128K; do not
   advertise 1M until long-context tests are passed.
7. Recheck upstream commits after v0.32.0 before every promotion; do not
   blindly overlay a moving version's attention backend.

All unimplemented or unverified cases must fail rather than silently
alter the model's math or claim a passing inference deployment.
