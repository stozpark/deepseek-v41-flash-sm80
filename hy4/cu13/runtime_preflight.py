#!/usr/bin/env python3
"""Reject wrong hardware/checkpoint before the expensive two-node load."""
import json
import os
from pathlib import Path
import torch


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def check_checkpoint(path, partition):
    config = json.loads((path / "config.json").read_text())
    t = config.get("text_config", config)
    require("HYV4VLForConditionalGeneration" in config.get("architectures", []),
            "not the flagship VL checkpoint")
    for name, expected in {
        "num_hidden_layers": 78, "hidden_size": 6144, "n_routed_experts": 256,
        "num_attention_heads": 64, "kv_lora_rank": 512, "qk_rope_head_dim": 64,
        "index_topk": 2048, "index_head_dim": 128, "index_n_heads": 32,
        "enable_ihc": True, "hc_mult": 4, "gated_mla": True,
        "learnable_sink": True, "num_nextn_predict_layers": 0,
        "swiglu_limit": 10.0, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "qk_nope_head_dim": 192,
        "v_head_dim": 256, "q_lora_rank": 2048,
    }.items():
        require(t.get(name) == expected, f"wrong checkpoint {name}: {t.get(name)!r}")
    q = config.get("quantization_config") or t.get("quantization_config") or {}
    require(q.get("quant_method") == "compressed-tensors", "expected compressed-tensors FP8")
    # Verify 128x128 block-FP8 scheme; permit other unquantized or separate
    # groups but never accept a compressed-tensors INT4/MXFP8 model by name alone.
    groups = q.get("config_groups") or {}
    require(any(
        g.get("weights", {}).get("num_bits") == 8
        and g.get("weights", {}).get("type") == "float"
        and g.get("weights", {}).get("block_structure") == [128, 128]
        for g in groups.values() if isinstance(g, dict) and isinstance(g.get("weights"), dict)
    ), "no 128x128 FP8 weight group in checkpoint")
    sizes = [int(x) for x in partition.split(",")]
    require(len(sizes) == 2 and min(sizes) > 0 and sum(sizes) == 78,
            "PP partition must contain two positive counts totaling 78")
    idx = t.get("indexer_types", [])
    require(idx == ["full" if i == 0 or (i-1) % 4 == 0 else "shared"
                    for i in range(78)], "wrong full/shared indexer pattern")
    require(t.get("layer_types") == ["deepseek_sparse_attention"] * 78,
            "wrong sparse attention layer contract")
    if idx[sizes[0]] == "shared":
        print("WARNING: PP stage 2 starts with shared indices; cross-stage transfer is unverified")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    shards = set(index.get("weight_map", {}).values())
    require(len(shards) == 245, f"expected 245 official shards, got {len(shards)}")
    require(all(Path(s).name == s and (path / s).is_file()
                and (path / s).stat().st_size > 0 for s in shards),
            "missing, empty, or invalid shard path")
    print("HY4_CHECKPOINT_METADATA=PASS (not a checksum/weight-integrity proof)")


def main():
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.version.cuda and torch.version.cuda.startswith("13.2"),
            f"wrong PyTorch CUDA runtime {torch.version.cuda}; source pin builds cu132")
    require(torch.cuda.device_count() == 8, "expected 8 visible GPUs")
    require("sm_80" in torch.cuda.get_arch_list(), "PyTorch wheel has no SM80 SASS")
    for i in range(8):
        p = torch.cuda.get_device_properties(i)
        require(torch.cuda.get_device_capability(i) == (8, 0), f"GPU {i}: not SM80")
        require(p.total_memory >= 75 * 1024**3, f"GPU {i}: not 80GB class")
        # Exercise allocation and an actual BF16 matrix multiply per GPU.
        with torch.cuda.device(i):
            a = torch.ones(16, 16, dtype=torch.bfloat16, device=f"cuda:{i}")
            require(float((a @ a)[0, 0]) == 16.0, f"GPU {i}: BF16 smoke failed")
            torch.cuda.synchronize(i)
    check_checkpoint(Path(os.environ["MODEL_PATH"]), os.environ["VLLM_PP_LAYER_PARTITION"])
    print("HY4_RUNTIME_PREFLIGHT=PASS", torch.__version__, torch.version.cuda)


if __name__ == "__main__":
    main()
