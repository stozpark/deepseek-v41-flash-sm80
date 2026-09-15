# Fully offline CUDA 13 SIF build

The production SM80 patch is already committed under `patches/generated/`; the disconnected build host does not need GitHub access. `patches/vendor/` is retained only for audit/regeneration and is not injected into the production SIF.

## Recommended: transfer a CUDA 13 base SIF

On an internet-connected machine:

```bash
apptainer build deepseek-v41-official-cu130-base.sif \
  docker://vllm/vllm-openai:deepseekv41-flash-0909
```

Transfer this repository and the base SIF to the disconnected A100 host. Then:

```bash
git checkout main
./prepare_sm80_overlay.sh
BASE_SIF=/data/deepseek-v41-official-cu130-base.sif ./build_sif.sh
```

Output:

```text
deepseek-v41-flash-sm80-cu130.sif
```

`prepare_sm80_overlay.sh` verifies the production-minimal 25-file patch and SHA256. `build_sif.sh` validates that the supplied base and completed SIF use a CUDA 13.0 PyTorch runtime.

This path performs no registry, GitHub, curl, or git network access.

## Local Docker archive

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909
docker save -o deepseekv41-cu130.tar vllm/vllm-openai:deepseekv41-flash-0909
```

After transfer:

```bash
BASE_URI=docker-archive:///data/deepseekv41-cu130.tar ./build_sif.sh
```

`BASE_SIF` remains the most portable offline method across Apptainer/Singularity versions.

## Verify production patch integrity

```bash
cd patches/generated
sha256sum -c sm80-cu130-minimal.patch.sha256
cat sm80-cu130-minimal.manifest.txt
```

The manifest must report `changed_files=25`.
