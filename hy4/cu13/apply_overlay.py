#!/usr/bin/env python3
"""Fail-closed overlay for the inspected vLLM #68633 head."""
import argparse
import ast
from pathlib import Path
import subprocess

PIN = "3470b0db1820f699960d1376223602739a0f4ac3"
HERE = Path(__file__).resolve().parent


def replace_once(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected exactly one source anchor")
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    src = args.source.resolve()
    head = subprocess.check_output(
        ["git", "-C", str(src), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PIN:
        raise RuntimeError(f"expected {PIN}, got {head}")
    model = src / "vllm/models/hy_v4/nvidia"
    attention = model / "attention.py"
    a = attention.read_text()
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
        model_path.read_text(),
        "buffers = {k: v[: positions.shape[0]] for k, v in self.pp_buffers.items()}",
        "buffers = {k: v[:full_num_tokens] for k, v in self.pp_buffers.items()}",
        "PP M-RoPE token axis: positions[-1] not positions[0]",
    )
    changes = {
        attention: a, model_path: m,
        model / "sm80_sparse.py": (HERE / "sm80_sparse.py").read_text(),
        model / "sm80_indexer_quant.py": (HERE / "sm80_indexer_quant.py").read_text(),
    }
    for path, data in changes.items():
        ast.parse(data, filename=str(path))
    if not args.check_only:
        for path in changes:
            if path.name.startswith("sm80_") and path.exists():
                raise RuntimeError(f"refusing to overwrite {path}")
        for path, data in changes.items():
            path.write_text(data)
    print(f"HYV4_SM80_OVERLAY={'CHECKED' if args.check_only else 'APPLIED'}")


if __name__ == "__main__":
    main()
