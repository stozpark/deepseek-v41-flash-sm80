#!/usr/bin/env python3
"""Stage5: correctly shaped SM80 paged-MQA for DSpark5 target verification.

vLLM 0.32.0's native speculative indexer supplies seq_lens as (B, next_n),
with one effective, exclusive compressed-cache end for *each* draft row.
Older SM80 code erroneously uses only the final length for every row,
which changes the causal mask and may leak future cache values.

This port enables next_n<=6 (one target + 5 DSpark drafts), preserves every
row's own end and refuses unverified DCP, PCP and >6 draft positions.
The CPU source replay is NOT a substitute for physical A100 correctness.
"""
from __future__ import annotations

import argparse
import compileall
import importlib.metadata
import shutil
from pathlib import Path

VLLM = "0.32.0"
IDX_REL = "model_executor/layers/sparse_attn_indexer.py"
PAGED_REL = "v1/attention/ops/mqa_logits_triton.py"


def replace_once(source: str, old: str, new: str, filename: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{filename}: expected one exact stage3 anchor, got {count}: {old[:120]!r}")
    return source.replace(old, new)


def prepare(root: Path) -> dict[Path, str]:
    idx_path, paged_path = root / IDX_REL, root / PAGED_REL
    indexer = idx_path.read_text(encoding="utf8")
    paged = paged_path.read_text(encoding="utf8")

    # Fail closed: these must be the stage3 SM80 indexer and its stage1
    # vendored Triton fallback, not an arbitrary future upstream revision.
    for value in (
        "def _sm80_indexer_fallback()",
        "SM80 plain one-token decode only (no DCP/PCP/spec)",
        "fp8_paged_mqa_logits_triton(",
    ):
        if value not in indexer:
            raise RuntimeError(f"Expected stage3 SM80 indexer anchor: {value}")
    for value in (
        "context_len = tl.load(context_lens_ptr + batch_id)",
        "q_offset = context_len - next_n + next_n_id",
        "B, next_n, num_heads, head_dim = q.shape",
    ):
        if value not in paged:
            raise RuntimeError(f"Expected stage1 SM80 paged kernel anchor: {value}")

    # Every output row must use its own causal position, including ragged
    # target batches, not a single sequence end shared by all draft rows.
    paged = replace_once(
        paged,
        "context_len = tl.load(context_lens_ptr + batch_id)",
        "context_len = tl.load(context_lens_ptr + token_id)",
        PAGED_REL,
    )
    paged = replace_once(
        paged,
        "q_offset = context_len - next_n + next_n_id",
        "q_offset = context_len - 1",
        PAGED_REL,
    )
    paged = replace_once(
        paged,
        "        context_lens:  [B] int32",
        "        context_lens:  [B, next_n] int32 per-query ends (or [B] final ends)",
        PAGED_REL,
    )
    # Support legacy single-token [B], legacy final-end [B] with next_n>1,
    # and true per-row [B, next_n] provided by 0.32.0. Per-query tails are
    # never computed from the last length when 2D metadata is supplied.
    anchor = "    B, next_n, num_heads, head_dim = q.shape\n"
    paged = replace_once(
        paged, anchor, anchor +
        """    if context_lens.dtype != torch.int32:
        raise TypeError("SM80 paged MQA expects int32 exclusive cache ends")
    if context_lens.ndim == 2:
        if tuple(context_lens.shape) != (B, next_n):
            raise ValueError(
                f"SM80 context_lens must be ({B},{next_n}); got {tuple(context_lens.shape)}"
            )
        row_lens = context_lens.contiguous().view(-1)
    elif context_lens.ndim == 1 and context_lens.numel() == B:
        if next_n == 1:
            row_lens = context_lens.contiguous()
        else:
            offsets = torch.arange(
                next_n, device=context_lens.device, dtype=torch.int32
            )
            row_lens = (
                context_lens[:, None] - next_n + 1 + offsets
            ).clamp_min(0).contiguous().view(-1)
    else:
        raise ValueError("SM80 paged MQA needs [B] or [B,next_n] sequence lengths")
    if not 1 <= next_n <= 6:
        raise ValueError(f"Unverified SM80 speculative verify width {next_n}")
""", PAGED_REL
    )
    paged = replace_once(
        paged,
        "        context_lens,\n        block_tables,\n        logits,\n",
        "        row_lens,\n        block_tables,\n        logits,\n",
        PAGED_REL,
    )
    # The older stale docs said that a single context end was shared across
    # all speculative rows. Replace this misleading statement.
    paged = replace_once(
        paged,
        "    # The block table is allocated at full max_model_len width and only\n",
        "    # Each query uses its own compressed-cache length; the per-row\n"
        "    # causal mask excludes uncommitted future speculative tokens.\n"
        "    # The block table is allocated at full max_model_len width and only\n",
        PAGED_REL,
    )

    indexer = replace_once(
        indexer,
        '            if next_n != 1 or dcp_world_size != 1 or use_pcp:\n'
        '                raise RuntimeError("SM80 plain one-token decode only (no DCP/PCP/spec)")',
        '            if dcp_world_size != 1 or use_pcp:\n'
        '                raise RuntimeError("SM80 DSpark5 indexer does not support DCP/PCP")\n'
        '            if next_n < 1 or next_n > 6:\n'
        '                raise RuntimeError("SM80 DSpark5 supports at most six target rows")',
        IDX_REL,
    )
    indexer = replace_once(
        indexer,
        "                seq_lens.reshape(-1).contiguous(),\n"
        "                decode_metadata.block_table,\n"
        "                max_model_len=max_model_len,\n"
        "                clean_logits=False,\n",
        "                seq_lens,\n"
        "                decode_metadata.block_table,\n"
        "                max_model_len=max_model_len,\n"
        "                clean_logits=(next_n > 1),\n",
        IDX_REL,
    )
    # clean_logits=True on spec queries initializes masked padded rows to
    # -inf, avoiding reading uninitialized values from the top-k path.
    # The legacy single-query version remains allocation-light.
    for path, source in ((idx_path, indexer), (paged_path, paged)):
        compile(source, str(path), "exec")
    return {idx_path: indexer, paged_path: paged}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path)
    p.add_argument("--skip-version-check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not args.skip_version_check and importlib.metadata.version("vllm") != VLLM:
        raise RuntimeError(f"Expected pinned vLLM {VLLM}")
    if args.source_root:
        root = args.source_root.resolve()
    else:
        import vllm
        root = Path(vllm.__file__).resolve().parent

    files = prepare(root)
    print("STAGE5_SM80_DSPARK_MULTITOKEN_SOURCE_PLAN_PASS", len(files))
    if args.dry_run:
        return

    backups: dict[Path, Path] = {}
    created: list[Path] = []
    try:
        for path, source in files.items():
            backup = path.with_name(path.name + ".sm80-v0320.pre-dspark")
            if backup.exists():
                raise RuntimeError(f"Refusing to overwrite prior stage5 backup {backup}")
            shutil.copy2(path, backup)
            backups[path] = backup
            path.write_text(source, encoding="utf8")
            created.append(path)
        for path in files:
            if not compileall.compile_file(str(path), quiet=1, force=True):
                raise RuntimeError(f"Stage5 patch failed to compile: {path}")
    except Exception:
        for path in created:
            shutil.move(str(backups[path]), str(path))
        raise
    print("STAGE5_SM80_DSPARK_SOURCE_PASS_NOT_GPU_VALIDATED")


if __name__ == "__main__":
    main()
