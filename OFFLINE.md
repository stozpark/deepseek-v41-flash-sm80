# Fully offline CUDA 12.9 SIF build

The SM80 patch sources are already committed in `patches/vendor/`.

On an internet-connected machine:

```bash
apptainer build deepseek-v41-official-cu129-base.sif docker://vllm/vllm-openai:deepseekv41-flash-0909-cu129
```

Transfer this repository plus the base SIF to the disconnected A100 host, then run:

```bash
BASE_SIF=/data/deepseek-v41-official-cu129-base.sif ./build_sif.sh
```

Output: `deepseek-v41-flash-sm80-cu129.sif`.

No GitHub, registry, curl, or git network access is used on this path.

Alternatively transfer a Docker archive and set `BASE_URI=docker-archive:///data/deepseekv41-cu129.tar` if your Apptainer/Singularity version supports that URI.
