#!/usr/bin/env python3
from __future__ import annotations

import compileall
import shutil
from pathlib import Path

import vllm

SRC = Path("/opt/sm80-overlay/vllm")
DST = Path(vllm.__file__).resolve().parent

required_base = [
    DST / "models/deepseek_v4_1",
    DST / "model_executor/kernels/linear/mxfp4/marlin.py",
    DST / "model_executor/kernels/linear/mxfp8/emulation.py",
]
missing_base = [str(p) for p in required_base if not p.exists()]
if missing_base:
    raise SystemExit(
        "Official DeepSeek-V4.1 image is not compatible with this overlay; "
        f"missing baseline paths: {missing_base}"
    )

if not SRC.is_dir():
    raise SystemExit(f"SM80 overlay missing: {SRC}")

copied = 0
for src in sorted(SRC.rglob("*")):
    if not src.is_file():
        continue
    rel = src.relative_to(SRC)
    dst = DST / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied += 1


def replace_once(path: Path, old: str, new: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if old not in text:
        raise SystemExit(
            f"{description}: expected baseline text not found in {path}; "
            "refusing an unsafe patch"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# Dependency closure for the SM80 sparse-indexer query/decode sharding added
# by wtdcode/vllm-backport a6ef07a + 6793ead.  The official 0909 image predates
# these two shared helpers, while the pinned SM80 indexer imports them.
# Keep the official distributed/utils.py otherwise untouched.
dist_utils = DST / "distributed/utils.py"
dist_text = dist_utils.read_text(encoding="utf-8")
if "def balanced_row_counts(" not in dist_text:
    dist_text += '''\n\n# SM80 backport compatibility: shared balanced TP row partition.\ndef balanced_row_counts(num_rows: int, size: int) -> list[int]:\n    """Per-rank row counts using base + (rank < remainder)."""\n    base, rem = divmod(num_rows, size)\n    return [base + (rank < rem) for rank in range(size)]\n\n\ndef balanced_row_bounds(\n    start: int, stop: int, rank: int, size: int\n) -> tuple[int, int]:\n    """This rank's contiguous half-open balanced range in [start, stop)."""\n    base, rem = divmod(stop - start, size)\n    lo = start + rank * base + min(rank, rem)\n    return lo, lo + base + (rank < rem)\n'''
    dist_utils.write_text(dist_text, encoding="utf-8")
elif "def balanced_row_bounds(" not in dist_text:
    raise SystemExit(
        "distributed.utils has balanced_row_counts but not balanced_row_bounds; "
        "unexpected baseline, refusing unsafe patch"
    )

# Post-0909 correctness hotfix: vllm-project/vllm#56297 / backport #78.
# Responses API uses input_text/output_text while the 0909 V4.1 tokenizer
# accepts only `text`.
tokenizer = DST / "tokenizers/deepseek_v41.py"
replace_once(
    tokenizer,
    'if part_type == "text":\n                    parts.append(block.get("text", ""))',
    'if part_type in ("text", "input_text", "output_text"):\n'
    '                    parts.append(block.get("text", ""))',
    "DeepSeek-V4.1 Responses API text-content hotfix",
)

# Post-0909 fix corresponding to vLLM PR #56623.  A V4.1 vision-capable
# checkpoint served with --language-model-only must not reserve the extra
# vision-visible SWA width (1024 tokens for V4.1-Flash).  This is useful on
# SM80 too: it avoids unnecessary metadata/index bandwidth.  If multimodal
# config is absent or language_model_only is false, behavior is unchanged.
sparse_swa = DST / "v1/attention/backends/mla/sparse_swa.py"
replace_once(
    sparse_swa,
    '''        self.max_image_tokens = (\n            getattr(hf_config, "vision_max_n_token", 0)\n            if getattr(hf_config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
    '''        mm_config = getattr(self.vllm_config.model_config, "multimodal_config", None)\n        language_model_only = bool(getattr(mm_config, "language_model_only", False))\n        self.max_image_tokens = (\n            0\n            if language_model_only\n            else (\n                getattr(hf_config, "vision_max_n_token", 0)\n                if getattr(hf_config, "vision_n_layers", 0) > 0\n                else 0\n            )\n        )''',
    "DeepSeek-V4.1 language-model-only SWA-width hotfix",
)

v41_attention = DST / "models/deepseek_v4_1/attention.py"
replace_once(
    v41_attention,
    '''        self.max_image_tokens = (\n            getattr(config, "vision_max_n_token", 0)\n            if getattr(config, "vision_n_layers", 0) > 0\n            else 0\n        )''',
    '''        mm_config = getattr(vllm_config.model_config, "multimodal_config", None)\n        language_model_only = bool(getattr(mm_config, "language_model_only", False))\n        self.max_image_tokens = (\n            0\n            if language_model_only\n            else (\n                getattr(config, "vision_max_n_token", 0)\n                if getattr(config, "vision_n_layers", 0) > 0\n                else 0\n            )\n        )''',
    "DeepSeek-V4.1 attention language-model-only SWA-width hotfix",
)

required_overlay = [
    DST / "v1/attention/ops/fp8_sm80.py",
    DST / "v1/attention/ops/mqa_logits_triton.py",
    DST / "models/deepseek_v4_1/ampere/ampere_sparse.py",
    DST / "v1/attention/backends/mla/indexer.py",
]
missing_overlay = [str(p) for p in required_overlay if not p.exists()]
if missing_overlay:
    raise SystemExit(f"SM80 overlay incomplete after copy: {missing_overlay}")

# Long-context correctness invariants from the SM80 paged-indexer fallback.
mqa_text = (DST / "v1/attention/ops/mqa_logits_triton.py").read_text(encoding="utf-8")
for required in (".to(tl.int64)", "k_offset < context_len"):
    if required not in mqa_text:
        raise SystemExit(
            "SM80 paged MQA logits is missing the long-context correctness fix: "
            + required
        )

if not compileall.compile_dir(str(DST), quiet=1, force=False):
    raise SystemExit("Python compile check failed after SM80 overlay")

marker = Path("/opt/sm80-overlay/APPLIED")
marker.write_text(
    f"target={DST}\nfiles={copied}\n"
    "balanced_row_helpers=1\n"
    "responses_api_text_hotfix=1\n"
    "language_model_only_swa_hotfix=1\n"
    "mqa_long_context_guards=1\n",
    encoding="utf-8",
)
print(f"Applied {copied} SM80 overlay files to {DST}")
print("Applied SM80 dependency-closure and post-0909 guarded hotfixes")
