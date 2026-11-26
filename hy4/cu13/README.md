# HY4 FP8 SM80 / CUDA 13 — experimental port

Target: public tencent/Hy4-FP8, NOT Flash or Preview, 2 nodes x 8 A100 80GB.
Source pin: vLLM PR #68633 head 3470b0db1820f699960d1376223602739a0f4ac3
(inspected 2026-11-26), including flagship registration, Triton sparse MLA/MQA
and software E4M3FN. This is an unmerged PR, not a stable release.

## Implementation

- Model-specific SM80 sink-aware sparse MLA with natural-log LSE and a
  learned zero-value sink; no sink-dropping dense prefill.
- Software E4M3FN query encoding and existing unfused RoPE. Reuse generic
  Triton MQA/paged MQA from #68633.
- PP shared top-k slicing on the correct M-RoPE token axis. Upstream already
  implements pp_topk_buffers/copy_pp_buffers.
- Source inspection confirms compressed-tensors W8A8 linear dispatch to
  W8A16 on SM80, including block-FP8 scales for Marlin. MoE oracle supports
  explicit marlin with 128x128 blocks. These paths still need GPU testing.
- Official checkpoint, Gated MLA, iHC, router and parsers retained.
- Fail-closed source overlay, build, initial two-node launcher, GPU tests.

## Build

At this pin the upstream Dockerfile defaults to CUDA **13.2.1** and includes
SM80 in its architecture list. A generic CUDA 13 environment does not prove
driver compatibility with 13.2.1. If host requires CUDA 13.0 specifically,
resolve a coherent alternate source/build pin. Do not just override CUDA_VERSION
while leaving the CUDA 13.2 builder and dependency lock unchanged.

The official hy4 Docker tag is mutable and may carry a different native ABI.
We build native extensions together with the source instead of blindly mixing
a newer Python overlay with that image.

On a connected Docker/BuildKit host:

    cd hy4/cu13
    bash prepare_source.sh
    bash build.sh

Optional: SIF_PATH=/path/hy4-sm80-cu13.sif bash build.sh
(Apptainer conversion from the local Docker daemon; not yet tested).
Record image ID, driver, checkpoint revision and hashes before transferring
identical immutable images/checkpoints to both nodes.

Run GPU tests before loading the full checkpoint:

    docker run --rm --gpus all --ipc=host -v "$PWD:/port:ro" \
      --entrypoint python3 hy4-sm80:cu13-dev /port/test_kernels.py

Apptainer: apptainer exec --nv /path/hy4-sm80-cu13.sif python3 test_kernels.py

## Two-node candidate (not validated)

Both nodes must have eight visible A100s, same model revision/local path,
same image, adequate RAM/disk, and routable master IP/port. Mount this
directory and checkpoint in the container. Use cluster-specific networking
and NCCL interfaces; do not expose the API publicly. Node 0:

    MODEL_PATH=/models/Hy4-FP8 MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash serve.sh

Node 1:

    MODEL_PATH=/models/Hy4-FP8 MASTER_ADDR=10.0.0.1 NODE_RANK=1 bash serve.sh

Replace example IP. Worker is headless. TP8 x PP2, 37/41 layers (stage 2
starts on a full indexer), eager, text-only, BF16 main KV, prefix cache off,
4K context, one request. DFly2 off because PP is unsupported with it.
Both nodes start with identical options except node rank. After passing
this profile, test 39/39 to exercise cross-stage shared top-k, then 32K,
128K, concurrency, prefix cache, graph and vision. Full 1M requires more
KV capacity; do not equate configured length with a successful request.

## Outstanding validation

No Docker build, GPU kernel execution or full-checkpoint inference has been
performed by this authoring session. Source anchors checked against pinned
GitHub files. The source pin is an unmerged community PR; GPU regressions
are possible. Test K-cache insertion/gather, FP8 Marlin repacking and MoE,
iHC, gated MLA, actual model loading, PP transfer, prompt/response equivalence,
tool calls, then long context. No claim of A100 serving success or speed.
See ../PORTING_PLAN.md for external evidence and validation gates (older
CUDA 12.9 assumption superseded by this CUDA 13 plan).
