#!/usr/bin/env python3
"""Create a small official-architecture config, NOT a real checkpoint.

The fixture uses vLLM deterministic dummy loading, preserving iHC, Gated MLA,
MLA dims, 64 heads, top-2048, top-8 and a shared indexer across PP 2/2. It
cannot validate real checkpoint weights or quality. Copy tokenizer assets only.
"""
import argparse
import json
import shutil
from pathlib import Path


def fixture_config(cfg, fp8=False):
    # Independent deep copy: preserve the source checkpoint config.
    cfg=json.loads(json.dumps(cfg))
    assert "HYV4VLForConditionalGeneration" in cfg["architectures"]
    updates=dict(num_hidden_layers=4,hidden_size=512,intermediate_size=1024,
                 q_lora_rank=256,n_routed_experts=16,
                 max_position_embeddings=8192,first_k_dense_replace=1,
                 indexer_types=["full","full","shared","shared"],
                 mlp_layer_types=["dense","sparse","sparse","sparse"],
                 layer_types=["deepseek_sparse_attention"]*4)
    for target in (cfg,cfg["text_config"]):
        target.update(updates)
        for key in list(target):
            if key.startswith("dfly_"):
                del target[key]
        if not fp8:
            target.pop("quantization_config",None)
    return cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint",type=Path)
    p.add_argument("destination",type=Path)
    p.add_argument("--fp8",action="store_true")
    a=p.parse_args()
    cfg=fixture_config(json.loads((a.checkpoint/"config.json").read_text()),a.fp8)
    a.destination.mkdir(parents=True,exist_ok=False)
    for name in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json",
                 "chat_template.jinja","generation_config.json","tokenizer.model",
                 "vocab.json","merges.txt","added_tokens.json","preprocessor_config.json",
                 "processor_config.json","image_processor_config.json"):
        path=a.checkpoint/name
        if path.is_file():
            shutil.copyfile(path,a.destination/name)
    if not (a.destination/"tokenizer_config.json").exists():
        raise RuntimeError("official tokenizer_config.json missing")
    (a.destination/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
    (a.destination/"HY4_DUMMY_FIXTURE").write_text("fp8\n" if a.fp8 else "bf16\n")
    print("HY4_DUMMY_FIXTURE_CREATED",a.destination)


if __name__=="__main__":
    main()
