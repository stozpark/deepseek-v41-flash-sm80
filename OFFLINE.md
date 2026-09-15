# Fully offline SIF build

The SM80 patch sources are already committed in `patches/vendor/`. The build
host does not need GitHub access.

## Recommended: transfer a base SIF

On an internet-connected machine with Apptainer/Singularity:

```bash
apptainer build deepseek-v41-official-cu130-base.sif \
  docker://vllm/vllm-openai:deepseekv41-flash-0909
```

Transfer both this repository and the base SIF to the disconnected A100 host.
Then:

```bash
BASE_SIF=/data/deepseek-v41-official-cu130-base.sif ./build_sif.sh
```

Output:

```text
deepseek-v41-flash-sm80-cu130.sif
```

This path performs no registry, GitHub, curl, or git network access.

## Alternative: local Docker archive / daemon

On a connected machine:

```bash
docker pull vllm/vllm-openai:deepseekv41-flash-0909
docker save -o deepseekv41-cu130.tar \
  vllm/vllm-openai:deepseekv41-flash-0909
```

After transferring the tar file, recent Apptainer versions can use a local
archive URI:

```bash
BASE_URI=docker-archive:///data/deepseekv41-cu130.tar ./build_sif.sh
```

If the image has already been loaded into a local Docker daemon, a
`docker-daemon://...` URI can be supplied through `BASE_URI` instead. URI
support varies by Apptainer/Singularity version, so `BASE_SIF` is the most
portable offline method.

## Verify vendored patch integrity

```bash
cd patches/vendor
sha256sum -c SHA256SUMS
```
