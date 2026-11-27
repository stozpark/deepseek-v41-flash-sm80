#!/usr/bin/env python3
"""Fail-closed overlay for the inspected vLLM #68633 head."""
import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

PIN = "3470b0db1820f699960d1376223602739a0f4ac3"
HERE = Path(__file__).resolve().parent


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one source anchor")
    return text.replace(old, new, 1)


def git(src, *args):
    return subprocess.check_output(["git", "-C", str(src), *args])


def pinned_text(src, path, expected_sha):
    rel = path.relative_to(src).as_posix()
    raw = git(src, "show", f"{PIN}:{rel}")
    digest = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(f"pinned source blob mismatch: {rel}")
    return raw.decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--verify-applied", action="store_true")
    args = ap.parse_args()
    src = args.source.resolve()
    head = git(src, "rev-parse", "HEAD").decode().strip()
    if head != PIN:
        raise RuntimeError(f"expected {PIN}, got {head}")
    if not args.verify_applied and git(src, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("expected clean pinned checkout")
    model = src / "vllm/models/hy_v4/nvidia"
    attention = model / "attention.py"
    a = pinned_text(src, attention, "f4b15ebdc9e3d88610986806ddd1d420e8766a55")
    a = replace_once(
        a,
        '            and self.scale_fmt == "ue8m0"\n'
        '            and current_platform.is_cuda()\n',
        '            and self.scale_fmt == "ue8m0"\n'
        '            and current_platform.is_cuda()\n'
        '            and not current_platform.is_device_capability(80)\n',
        "disable native fused indexer Q FP8 on SM80",
    )
    a = replace_once(
        a,
        """        q_fp8, q_scale = per_token_group_quant_fp8(
            q,
            self.quant_block_size,
            column_major_scales=False,
            use_ue8m0=self.scale_fmt is not None,
        )
""",
        """        if current_platform.is_cuda() and current_platform.is_device_capability(80):
            from .sm80_indexer_quant import quantize_indexer_q

            q_fp8, q_scale = quantize_indexer_q(
                q, self.quant_block_size, self.scale_fmt is not None
            )
        else:
            q_fp8, q_scale = per_token_group_quant_fp8(
                q,
                self.quant_block_size,
                column_major_scales=False,
                use_ue8m0=self.scale_fmt is not None,
            )
""",
        "software E4M3FN query quantization",
    )
    a = replace_once(
        a,
        """        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        try:
""",
        """        head_size = self.kv_lora_rank + self.qk_rope_head_dim
        dtype = torch.get_default_dtype()
        if current_platform.is_cuda() and current_platform.is_device_capability(80):
            from .sm80_sparse import HYV4SM80SparseBackend

            cfg = get_current_vllm_config()
            parallel = cfg.parallel_config
            if (
                cfg.speculative_config is not None
                or parallel.decode_context_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or cfg.attention_config.hisparse_config is not None
            ):
                raise RuntimeError("HYV4 SM80: speculation/PCP/DCP/HiSparse not validated")
            if not self.is_sparse or head_size != 576:
                raise RuntimeError("HYV4 SM80 requires sparse MLA with head size 576")
            if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
                raise RuntimeError("HYV4 SM80 requires unquantized main MLA KV")
            selected = get_attn_backend(
                head_size=head_size,
                dtype=dtype,
                kv_cache_dtype=kv_cache_dtype,
                use_mla=True,
                use_sparse=True,
                num_heads=self.num_local_heads,
            )
            if selected.get_name() != "TRITON_MLA_SPARSE":
                raise RuntimeError("HYV4 SM80 requires TRITON_MLA_SPARSE")
            cfg.attention_config.sparse_mla_shard_q = False
            cfg.attention_config.sparse_mla_force_mqa = True
            logger.info_once("HYV4 SM80: sink-aware Triton sparse MLA enabled")
            return HYV4SM80SparseBackend
        try:
""",
        "sink backend binding",
    )
    model_path = model / "model.py"
    m = replace_once(
        pinned_text(src, model_path, "8eebff472fbf4522752b6d9bda566c028024cb5a"),
        "buffers = {k: v[: positions.shape[0]] for k, v in self.pp_buffers.items()}",
        "buffers = {k: v[:full_num_tokens] for k, v in self.pp_buffers.items()}",
        "PP M-RoPE token axis: positions[-1] not positions[0]",
    )
    changes = {
        attention: a, model_path: m,
        model / "sm80_sparse.py": (HERE / "sm80_sparse.py").read_text(),
        model / "sm80_indexer_quant.py": (HERE / "sm80_indexer_quant.py").read_text(),
        model / "sink_math.py": (HERE / "sink_math.py").read_text(),
    }
    for path, data in changes.items():
        ast.parse(data, filename=str(path))
    new_paths = {p for p in changes if p not in (attention, model_path)}
    if args.verify_applied:
        expected = {p.relative_to(src).as_posix() for p in changes}
        tracked = git(src, "diff", "--name-only", "-z", "HEAD").decode().split("\0")
        untracked = git(src, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
        actual = set(tracked + untracked) - {""}
        if actual != expected:
            raise RuntimeError(f"source drift: extra={actual-expected}, missing={expected-actual}")
        if git(src, "diff", "--cached", "--name-only"):
            raise RuntimeError("refusing staged changes")
        for path, data in changes.items():
            if path.is_symlink() or path.read_bytes() != data.encode("utf-8"):
                raise RuntimeError(f"overlay mismatch: {path}")
        print("HYV4_SM80_OVERLAY=VERIFIED")
        return
    for path in (attention, model_path):
        if path.is_symlink() or path.read_bytes() != git(src, "show", f"{PIN}:{path.relative_to(src).as_posix()}"):
            raise RuntimeError(f"working tree differs from pin: {path}")
    for path in new_paths:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to overwrite {path}")
    if not args.check_only:
        for path, data in changes.items():
            path.write_bytes(data.encode("utf-8"))
    print(f"HYV4_SM80_OVERLAY={'CHECKED' if args.check_only else 'APPLIED'}")


if __name__ == "__main__":
    main()
