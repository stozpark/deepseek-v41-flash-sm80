# Fully offline CUDA 12.9 SIF build

The SM80 patch sources are already committed in `patches/vendor/`. The build
host does not need GitHub access.

## Recommended: transfer a CUDA 12.9 base SIF

On an internet-connected machine with Apptainer/Singularity:

```bash
apptainer build deepseek-v41-official-cu129-base.sif \
  docker://vllm/vllm-openai:deepseekv41-flash-0909-cu129
```

Transfer both this repository and the base SIF to the disconnected A100 host.
Then:

```bash
git checkout cu129
BASE_SIF=/data/deepseek-v41-official-cu129-base.sif ./build_sif.sh
```

Output:

```text
deepseek-v41-flash-sm80-cu129.sif
```

`build_sif.sh` checks `torch.version.cuda` inside the supplied base and refuses
to continue unless it is CUDA 12.9. It validates the completed SIF again after
build.

This path performs no registry, GitHub, curl, or git network access.

## Alternative: local Docker archive / daemon

On a connected machine:

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909-cu129
docker save -o deepseekv41-cu129.tar \
  vllm/vllm-openai:deepseekv41-flash-0909-cu129
```

After transferring the tar file:

```bash
BASE_URI=docker-archive:///data/deepseekv41-cu129.tar ./build_sif.sh
```

If the image has already been loaded into a local Docker daemon, a
`docker-daemon://...` URI can be supplied through `BASE_URI`. URI support varies
by Apptainer/Singularity version, so `BASE_SIF` is the most portable offline
method.

## Verify vendored patch integrity

```bash
cd patches/vendor
sha256sum -c SHA256SUMS
```
