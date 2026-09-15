# Fully offline CUDA 12.9 SIF build

The production SM80 patch is already committed under `patches/generated/`; the disconnected build host does not need GitHub access. `patches/vendor/` is retained only for audit/regeneration and is not injected into the production SIF.

## Recommended: transfer a CUDA 12.9 base SIF

On an internet-connected machine with Apptainer/Singularity:

```bash
apptainer build deepseek-v41-official-cu129-base.sif \
  docker://vllm/vllm-openai:deepseekv41-flash-0909-cu129
```

Transfer both this repository and the base SIF to the disconnected A100 host. Then:

```bash
git checkout cu129
./prepare_sm80_overlay.sh
BASE_SIF=/data/deepseek-v41-official-cu129-base.sif ./build_sif.sh
```

Output:

```text
deepseek-v41-flash-sm80-cu129.sif
```

`prepare_sm80_overlay.sh` now verifies the production-minimal 25-file patch and its SHA256; it does not create or copy a broad source overlay. `build_sif.sh` also verifies the patch, checks `torch.version.cuda` in the supplied base, and refuses to continue unless it is CUDA 12.9. The completed SIF is validated again.

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

If the image is already loaded into a local Docker daemon, a `docker-daemon://...` URI can be supplied through `BASE_URI`. URI support varies by Apptainer/Singularity version, so `BASE_SIF` is the most portable offline method.

## Verify production patch integrity

```bash
cd patches/generated
sha256sum -c sm80-cu129-minimal.patch.sha256
cat sm80-cu129-minimal.manifest.txt
```

The manifest must report `changed_files=25`.

For maintainer-side broad source audit only:

```bash
cd patches/vendor
sha256sum -c SHA256SUMS
```
