#!/usr/bin/env python3
"""Exact-source SM80 software FP8 conversion port for vLLM v0.32.0.

Uses software E4M3FN (RNE, saturate-finite) rather than forbidden SM89+
Triton fp8 instructions on A100. Does not attempt DeepGEMM-free sparse
indexer integration; stage3 is still needed before serving is supported.
Never claim an accuracy PASS based on successful source replacement alone.
"""
from __future__ import annotations

import argparse
import compileall
import hashlib
import importlib.metadata
import shutil
from pathlib import Path

EXPECTED_VLLM = "0.32.0"
ORIGINAL_SHA = {
    "models/deepseek_v41/common/ops/cache_utils.py": "a6139c71f836d931cd2f1850168243dcaa3fec30",
    "models/deepseek_v41/common/ops/fused_compress_quant_cache.py": "2ba9dbea3de7c66cfa3cb17dd74a06c80e0d46e5",
    "models/deepseek_v41/common/ops/indexer_k_store.py": "ba8ddbd95a9a627aa1938a47cc69f9ecf9a757f3",
    "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py": "dd25c73af7c2ce4fd30f179f8a476c06011e3b58",
}
ROCM_REL = "v1/attention/ops/rocm_aiter_mla_sparse.py"
BELOW_SM89_DIRECT_CAST = ".to(tl.float8e4nv"


def git_oid(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}".encode() + bytes([0]) + data).hexdigest()


def swap(src: str, old: str, new: str, expected: int, path: str) -> str:
    found = src.count(old)
    if found != expected:
        raise RuntimeError(f"{path}: expected {expected} occurrences of {old!r}; found {found}")
    return src.replace(old, new)


def add_import(src: str, import_line: str, path: str) -> str:
    return swap(src, "from vllm.triton_utils import tl, triton\n",
                import_line + "from vllm.triton_utils import tl, triton\n", 1, path)


def numeric_patch(root: Path) -> dict[Path, bytes]:
    updates: dict[Path, bytes] = {}
    for relative, expected in ORIGINAL_SHA.items():
        raw = (root / relative).read_bytes()
        got = git_oid(raw)
        if got != expected:
            raise RuntimeError(f"Refusing to change unpinned {relative}: {got} != {expected}")
    rocm = (root / ROCM_REL).read_text(encoding="utf8")
    if "    return output.shape[0] if fused_inv_rope else 0\n" not in rocm:
        raise RuntimeError("Stage1 upstream #62197 inverse-RoPE fix must be present")
    if "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32" in rocm:
        raise RuntimeError("Stage2 numeric patch appears already applied")
    rocm = add_import(
        rocm,
        "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32\n",
        ROCM_REL,
    )
    rocm = swap(
        rocm,
        "x_uint8.to(tl.float8e4nv, bitcast=True).to(tl.float32)",
        "_decode_fp8_f32(x_uint8, False)",
        2, ROCM_REL,
    )
    rocm = swap(
        rocm,
        "x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)",
        "x_fp8 = _decode_fp8_f32(x_uint8, False)",
        4, ROCM_REL,
    )
    # Asserts no remaining direct E4M3 casts in this reused Triton attention
    # module; any future cast must be audited before a version bump.
    assert BELOW_SM89_DIRECT_CAST not in rocm, ROCM_REL
    updates[root / ROCM_REL] = rocm.encode()

    rel = "models/deepseek_v41/common/ops/cache_utils.py"
    src = add_import(
        (root / rel).read_text(), 
        "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32, _encode_e4m3fn_u8\n",
        rel,
    )
    src = swap(src, 
        """            if use_fnuz:
                x_fp8 = x_clamped.to(tl.float8e4b8)
            else:
                x_fp8 = x_clamped.to(tl.float8e4nv)
            x_uint8 = x_fp8.to(tl.uint8, bitcast=True)""",
        """            if use_fnuz:
                x_uint8 = x_clamped.to(tl.float8e4b8).to(tl.uint8, bitcast=True)
            else:
                x_uint8 = _encode_e4m3fn_u8(x_clamped)""", 
        1, rel)
    src = swap(src,
        """    if use_fnuz:  # noqa: SIM108
        fp8 = scaled.to(tl.float8e4b8)
    else:
        fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        token_data_ptr + d, tl.reshape(fp8.to(tl.uint8, bitcast=True), (head_dim,))
    )""",
        """    if use_fnuz:
        packed = scaled.to(tl.float8e4b8).to(tl.uint8, bitcast=True)
    else:
        packed = _encode_e4m3fn_u8(scaled)
    tl.store(token_data_ptr + d, tl.reshape(packed, (head_dim,)))""",
        1, rel)
    src = swap(
        src, "x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)",
        "x_fp8 = _decode_fp8_f32(x_uint8, False)",
        3, rel,
    )
    src = swap(
        src, "sf.to(tl.float8e4nv, bitcast=True).to(tl.float32)",
        "_decode_fp8_f32(sf, False)",
        1, rel,
    )
    assert BELOW_SM89_DIRECT_CAST not in src, rel
    updates[root / rel] = src.encode()

    rel = "models/deepseek_v41/common/ops/fused_compress_quant_cache.py"
    src = add_import(
        (root / rel).read_text(), 
        "from vllm.v1.attention.ops.fp8_sm80 import _decode_fp8_f32, _encode_e4m3fn_u8\n",
        rel,
    )
    src = swap(src,
        """    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    packed = tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,))""",
        """    packed = tl.reshape(
        _encode_e4m3fn_u8(tl.clamp(scaled, -448.0, 448.0)), (512,)
    )""",
        1, rel)
    src = swap(src,
        """    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(values + d, tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,)))""",
        """    fp8_bytes = _encode_e4m3fn_u8(tl.clamp(scaled, -448.0, 448.0))
    tl.store(values + d, tl.reshape(fp8_bytes, (512,)))""",
        1, rel)
    # NVFP4 compressed KV needs E4M3 scale rounding before FP4 act
    # quantization; raw int8 cast would change arithmetic and is forbidden.
    src = swap(src,
        """    scale = tl.clamp(amax * (1.0 / 6.0), 0.001953125, 448.0).to(tl.float8e4nv)
    # Round-to-nearest division: Triton's default div.full misplaces values
    # that land exactly on an e2m1 tie.
    scaled = tl.math.div_rn(tiles, tl.reshape(scale.to(tl.float32), (32, 1)))""",
        """    scale_bytes = _encode_e4m3fn_u8(
        tl.clamp(amax * (1.0 / 6.0), 0.001953125, 448.0)
    )
    # Preserve FP8-rounded scale before the e2m1 quantization division.
    scale_float = _decode_fp8_f32(scale_bytes, False)
    scaled = tl.math.div_rn(tiles, tl.reshape(scale_float, (32, 1)))""",
        1, rel)
    src = swap(src,
        "    tl.store(scales + tl.arange(0, 32), scale.to(tl.uint8, bitcast=True))",
        "    tl.store(scales + tl.arange(0, 32), scale_bytes)", 1, rel)
    src = swap(src,
        """        tl.store(dst + d, tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv))""",
        """        tl.store(
            dst.to(tl.pointer_type(tl.uint8)) + d,
            _encode_e4m3fn_u8(tl.clamp(scaled, -448.0, 448.0)),
        )""",
        1, rel)
    assert BELOW_SM89_DIRECT_CAST not in src, rel
    updates[root / rel] = src.encode()

    rel = "models/deepseek_v41/common/ops/indexer_k_store.py"
    src = add_import(
        (root / rel).read_text(), 
        "from vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8\n",
        rel,
    )
    src = swap(src,
        "x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)",
        "_encode_e4m3fn_u8(x_clamped)", 1, rel,
    )
    assert BELOW_SM89_DIRECT_CAST not in src, rel
    updates[root / rel] = src.encode()

    rel = "models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py"
    src = add_import(
        (root / rel).read_text(), 
        "from vllm.v1.attention.ops.fp8_sm80 import _encode_e4m3fn_u8\n",
        rel,
    )
    src = swap(src,
        """        x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)""",
        """        x_quant_u8 = _encode_e4m3fn_u8(tl.clamp(x / scales_exp, -fp8_max, fp8_max))""",
        1, rel)
    src = swap(src,
        "        tl.store(out_base + offsets, x_quant)",
        "        tl.store(out_base.to(tl.pointer_type(tl.uint8)) + offsets, x_quant_u8)",
        1, rel)
    assert BELOW_SM89_DIRECT_CAST not in src, rel
    updates[root / rel] = src.encode()
    return updates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-version-check", action="store_true",
                        help="CI source test only; pinned exact source is still required")
    args = parser.parse_args()
    if not args.skip_version_check:
        actual = importlib.metadata.version("vllm")
        if actual != EXPECTED_VLLM:
            raise RuntimeError(f"vLLM version mismatch {actual} != {EXPECTED_VLLM}")
    if args.vllm_root:
        root = args.vllm_root.resolve()
    else:
        import vllm
        root = Path(vllm.__file__).resolve().parent
    updates = numeric_patch(root)
    print(f"STAGE2_SM80_FP8_SOURCE_PLAN_OK files={len(updates)}")
    if args.dry_run:
        return
    backups: list[Path] = []
    try:
        for file, content in updates.items():
            backup = file.with_name(file.name + ".sm80-v0320.stage1")
            if backup.exists():
                raise RuntimeError(f"Previous patch backup already exists: {backup}")
            shutil.copy2(file, backup)
            backups.append(backup)
            file.write_bytes(content)
        for file in updates:
            if not compileall.compile_file(str(file), quiet=1, force=True):
                raise RuntimeError(f"Syntax failure in {file}")
    except Exception:
        for backup in backups:
            shutil.move(str(backup), str(backup).removesuffix(".sm80-v0320.stage1"))
        raise
    print("STAGE2_SM80_FP8_SOURCE_PASS_NOT_GPU_VALIDATED")


if __name__ == "__main__":
    main()
