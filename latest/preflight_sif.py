#!/usr/bin/env python3
"""Pre-model-load check: source->SIF integrity, runtime ABI, checkpoint inventory.

No vLLM server starts and no model weights are allocated. Designed to catch
stale/renamed SIFs and missing shards; *cannot* prove A100 kernel correctness.
Only stdlib is required on the host. Use --source-only for GitHub Actions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPECTED_VLLM = "0.32.0"
EXPECTED_BASE = "vllm/vllm-openai:v0.32.0"
EXPECTED_SHA = "5df63a7203cdb9328d824a27b682e77c7b9928be"
MIN_CUDA_MAJOR = 13


def fatal(message: str) -> None:
    raise SystemExit("PRE-MODEL PREFLIGHT FAIL: " + message)


def cmd(argv: list[str], *, timeout: int = 120) -> str:
    try:
        run = subprocess.run(
            argv, capture_output=True, text=True, check=True, timeout=timeout
        )
    except (subprocess.CalledProcessError, FileNotFoundError,
            subprocess.TimeoutExpired) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = f"\nstdout={exc.stdout[-2500:]}\nstderr={exc.stderr[-2500:]}"
        fatal(f"command failed: {argv[:5]}: {exc}{detail}")
    return run.stdout


def read_env(path: Path) -> dict[str,str]:
    values: dict[str,str]={}
    for raw in path.read_text().splitlines():
        line=raw.strip()
        if line and not line.startswith("#"):
            match=re.fullmatch(r"([A-Z_]+)=([a-zA-Z0-9._/\-]+)",line)
            if not match:
                fatal(f"invalid pinned VERSION.env entry: {line!r}")
            values[match[1]]=match[2]
    return values


def source_manifest(root: Path) -> dict[str,str]:
    df=(root/"latest/Singularity.def").read_text()
    if "Bootstrap: docker\nFrom: "+EXPECTED_BASE not in df:
        fatal("Singularity.def no longer uses exact approved base image")
    labels={"UpstreamCommit":EXPECTED_SHA, "VLLMVersion":EXPECTED_VLLM}
    for key,value in labels.items():
        if not re.search(r"(?m)^  "+key+r" "+re.escape(value)+r"$",df):
            fatal(f"definition label {key} differs from pinned source")
    lines=df.splitlines()
    section=False
    mapped={}
    for line in lines:
        if line.strip()=="%files":
            section=True
            continue
        if line.startswith("%") and section:
            break
        if not section or not line.strip() or line.lstrip().startswith("#"):
            continue
        parts=line.split()
        if len(parts)!=2:
            fatal(f"invalid Singularity %files entry: {line!r}")
        src,dst=parts
        if dst in mapped:
            fatal(f"duplicate container destination: {dst}")
        file=(root/src).resolve()
        if root.resolve() not in file.parents or not file.is_file():
            fatal(f"missing or outside-root SIF input: {src}")
        mapped[dst]=hashlib.sha256(file.read_bytes()).hexdigest()
    if len(mapped)<18:
        fatal(f"too few source files copied into SIF: {len(mapped)}")
    for name in ("port_sm80_stage1.py","port_sm80_numeric.py",
                 "port_sm80_indexer.py","port_sm80_engram.py",
                 "port_sm80_dspark.py","test_sm80_fp8_pointer_abi.py",
                 "test_sm80_woa_contract.py"):
        if not any(x.endswith("/"+name) for x in mapped):
            fatal(f"required source/contract missing from image build: {name}")
    return mapped


def check_model(model_dir: Path, gpu_count: int) -> None:
    if not model_dir.is_dir():
        fatal(f"model directory missing: {model_dir}")
    cfgpath=model_dir/"config.json"
    if not cfgpath.is_file():
        fatal(f"model config missing: {cfgpath}")
    cfg=json.loads(cfgpath.read_text())
    if cfg.get("architectures")!=["DeepseekV41ForCausalLM"]:
        fatal(f"unexpected checkpoint architecture: {cfg.get('architectures')!r}")
    text=cfg.get("text_config") or cfg
    if int(text.get("o_groups",-1))!=8 or int(text.get("num_attention_heads",-1))!=64:
        fatal("1017 topology changed: expected 8 output groups, 64 attention heads")
    if gpu_count!=8:
        fatal("checkpoint test recipe requires exactly 8 visible GPUs per TP8 replica")

    index_files=list(model_dir.glob("*.safetensors.index.json"))
    if index_files:
        if len(index_files)>1:
            fatal("ambiguous safetensors index files")
        idx=json.loads(index_files[0].read_text())
        weights=idx.get("weight_map")
        if not isinstance(weights,dict) or not weights:
            fatal("invalid or empty safetensors weight map")
        shards=set(weights.values())
        missing=[name for name in shards if not (model_dir/name).is_file()]
        if missing:
            fatal(f"{len(missing)} missing weight shards: {missing[:8]}")
        corrupt=[name for name in shards if (model_dir/name).stat().st_size < 4096]
        if corrupt:
            fatal(f"empty/suspicious weight shards: {corrupt[:8]}")
        print(f"CHECKPOINT_SHARD_FILES_PASS count={len(shards)} tensors={len(weights)}")
    else:
        shards=list(model_dir.glob("*.safetensors"))
        if not shards:
            fatal("No safetensors weights found and no safetensors index")
        print(f"CHECKPOINT_SHARDS_WITHOUT_INDEX count={len(shards)} "
              "(cannot verify complete expected list)")
    for name in ("tokenizer.json","tokenizer_config.json"):
        if not (model_dir/name).is_file():
            print(f"WARNING missing {name}: verify separate tokenizer path")
    disk_bytes=sum(f.stat().st_size for f in shards if isinstance(f,Path)) if not index_files else sum((model_dir/n).stat().st_size for n in set(weights.values()))
    print(f"CHECKPOINT_BYTES_PRESENT={disk_bytes / (1024**3):.2f} GiB")

    n_rows=text.get("engram_num_embeddings")
    dim=text.get("engram_head_dim")
    if n_rows is not None and dim is not None:
        block=int(text.get("engram_block_size",32))
        nbytes=sum(map(int,n_rows))*(int(dim)+int(dim)//block)
        print(f"ESTIMATED_ENGRAM_HOST_RAM_TOTAL_TP8={nbytes/(1024**3):.2f} GiB "
              "(before extra buffers and OS, not per GPU)")
        meminfo=Path("/proc/meminfo")
        if meminfo.is_file():
            vals={}
            for line in meminfo.read_text().splitlines():
                m=re.match(r"(\w+):\s+(\d+)\s+kB",line)
                if m:
                    vals[m[1]]=int(m[2])*1024
            avail=vals.get("MemAvailable")
            if avail is not None:
                print(f"HOST_MEMORY_AVAILABLE={avail/(1024**3):.2f} GiB")
                if avail < nbytes * 1.1:
                    print("WARNING: available host RAM is below 110% of "
                          "estimated Engram tables; TP8 load may OOM")
    print("MODEL_MANIFEST_PREFLIGHT_PASS")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--source-only",action="store_true",
                        help="Validate SIF build inputs and cross-file source pins only")
    parser.add_argument("--sif",type=Path,help="Existing SIF file, checked bytewise")
    parser.add_argument("--model-dir",type=Path,help="1017 local checkpoint directory")
    parser.add_argument("--require-gpus",type=int,default=8)
    args=parser.parse_args()
    root=args.repo_root.resolve()
    config=read_env(root/"latest/VERSION.env")
    if config.get("VLLM_VERSION")!=EXPECTED_VLLM or config.get("VLLM_GIT_SHA")!=EXPECTED_SHA:
        fatal("VERSION.env release tag or source SHA drift")
    if config.get("VLLM_IMAGE")!=EXPECTED_BASE or int(config.get("CUDA_MIN_MAJOR","0"))<13:
        fatal("VERSION.env image/CUDA baseline drift")
    manifest=source_manifest(root)
    print(f"SOURCE_TO_SIF_MANIFEST_PASS sha256_files={len(manifest)}")
    if args.source_only:
        return
    if args.require_gpus!=8:
        fatal("single TP8 launcher supports exactly eight A100/A800 GPUs")
    runner=shutil.which("apptainer") or shutil.which("singularity")
    if not runner:
        fatal("apptainer/singularity executable missing on host")
    runner_version=cmd([runner,"--version"],timeout=15).strip()
    if not re.search(r"(apptainer|singularityce|singularity version|singularity-ce)",
                     runner_version,re.IGNORECASE):
        fatal(f"{runner} is not a recognized Apptainer/SingularityCE runtime: {runner_version}")
    print(f"CONTAINER_RUNNER_PASS {runner_version}")
    if args.sif is None:
        fatal("provide --sif after building; this preflight does NOT build")
    sif=args.sif.resolve()
    if not sif.is_file() or sif.stat().st_size < 1024*1024:
        fatal(f"missing/too-small SIF: {sif}")
    labels=cmd([runner,"inspect","--labels",str(sif)])
    for item in ("VLLMVersion","0.32.0","UpstreamCommit",EXPECTED_SHA):
        if item not in labels:
            fatal(f"SIF labels don't contain pinned {item}; stale/wrong image")
    print("SIF_VERSION_LABELS_PASS")
    targets=sorted(manifest)
    hashes=cmd([runner,"exec",str(sif),"sha256sum",*targets],timeout=180)
    observed={}
    for line in hashes.splitlines():
        m=re.match(r"^([0-9a-f]{64})\s+\*?(.+)$",line)
        if m:
            observed[m[2]]=m[1]
    changed=[name for name in targets if observed.get(name)!=manifest[name]]
    if changed:
        fatal(f"SIF differs from working tree or omits these files: {changed[:12]}")
    print(f"SIF_BYTE_FOR_BYTE_SOURCE_INTEGRITY_PASS {len(targets)} files")
    # No --nv: purely verify image's Python environment and patched source.
    validate=[
        "import importlib.metadata as md; import torch;"
        "assert md.version('vllm')=='0.32.0';"
        "assert int(str(torch.version.cuda or '0').split('.')[0])>=13;"
        "print('SIF_PYTHON_TORCH_CUDA_ABI_SOURCE_PASS',"
        "torch.__version__,torch.version.cuda)"
    ]
    print(cmd([runner,"exec",str(sif),"python3","-c",validate[0]],timeout=90).strip())
    print(cmd([runner,"exec",str(sif),"python3",
               "/opt/deepseek-sm80/audit_source.py","--mode","dspark"],timeout=90)[-1400:])
    if args.model_dir is not None:
        check_model(args.model_dir.resolve(),args.require_gpus)
    else:
        print("MODEL_MANIFEST_NOT_CHECKED (pass --model-dir)")
    print("PRE_MODEL_STATIC_IMAGE_PREFLIGHT_PASS "
          "(no A100 kernels or full-model inference validated)")


if __name__=="__main__":
    main()
