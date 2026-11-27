# HY4 FP8 SM80 / CUDA 13 — experimental port

Target: public tencent/Hy4-FP8, NOT Flash or Preview, 2 nodes x 8 A100 80GB.
Source pin: vLLM PR #68633 head 3470b0db1820f699960d1376223602739a0f4ac3
(pinned 2026-11-27), including flagship registration, Triton sparse MLA/MQA
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

On a connected Docker/BuildKit host (root-level DeepSeek scripts and CUDA 12.9 metadata are historical, not this port):

    cd hy4/cu13
    bash prepare_source.sh
    bash build.sh

Optional: SIF_PATH=/path/hy4-sm80-cu13.sif bash build.sh
(Apptainer conversion from the local Docker daemon; not yet tested).
Record image ID, driver, checkpoint revision and hashes before transferring
identical immutable images/checkpoints to both nodes.

The build now verifies exact overlay bytes, rejects unrelated source changes, and checks the packaged image's CUDA runtime and Hy4 modules. Native GPU compilation/runtime remain unverified. Run GPU tests before loading the full checkpoint:

    docker run --rm --gpus all --ipc=host -v "$PWD:/port:ro" \
      --entrypoint python3 hy4-sm80:cu132-dev /port/test_kernels.py

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
Both nodes start with identical options except node rank. The launcher checks all 8 visible GPUs, BF16 matmul, checkpoint architecture/FP8 scheme and shard existence, and runs the GPU kernel checks before the load on each node. This is not a cryptographic checkpoint verification; pin/hash weights and compare the same image ID or SIF SHA256 across nodes. After passing
this profile, test 39/39 to exercise cross-stage shared top-k, then 32K,
128K, concurrency, prefix cache, graph and vision. Full 1M requires more
KV capacity; do not equate configured length with a successful request.

## Review follow-up: mandatory validation sequence

The build creates image-manifest.json and checks native ABI imports, actual
module imports, file SHA256s and CLI arguments. Transfer that manifest and
the exact port directory with the immutable image/SIF to both nodes.
All launches now explicitly select Marlin linear and MoE, and audited_worker
checks every loaded rank's actual layer classes, sink aliases, indexer
pattern and repacked weights. Eager tracing prints HY4_SPARSE_FORWARD for
both prefill and decode on each layer/rank; preserve these logs.

Before full loading, run test_kernels.py (native Insert/Gather -> MQA ->
real Top-K -> physical indices -> sparse MLA, and 2/4GiB address crossing)
and test_marlin.py (actual FP8 repack/compute for TP8 local dimensions,
Top-8 MoE with active SwiGLU clamp). The full launcher runs both on each
node, not all kernels on all eight GPUs; BF16 preflight checks each GPU.

1. Build and verify image/SIF; run the GPU checks on both nodes.
2. Create a small BF16 fixture on shared storage:
       python3 make_fixture.py /models/Hy4-FP8 /models/hy4-fixture
   Optionally create a separate FP8 fixture with --fp8 for dispatch/packing
   smoke. Dummy load is NOT a real checkpoint or model-quality test.
3. Run the BF16 fixture with TP_SIZE=1 PP_SIZE=1 bash serve_fixture.sh on
   one GPU. In a separate shell capture:
       python3 serve_probe.py --output fixture-pp1.json --interleaved
   Then restart using TP_SIZE=1 PP_SIZE=2 NNODES=2 NODE_RANK=0/1 and
   MASTER_ADDR on the two nodes, one visible GPU each, 2,2 partition.
   Run serve_probe.py --output fixture-pp2.json --reference fixture-pp1.json.
   Require all worker dispatch/forward logs, nonempty logits, and matched
   tokens/logprobs. Repeat with 1,3 partition and a different batch budget.
   Repeat FP8 fixture for Marlin dispatch. Dummy weight initialization and
   PP arithmetic can differ; a mismatch is a failure to investigate, not
   a reason to weaken tolerance or declare parity.
4. Before full TP8 x PP2, verify actual NCCL fabric in the same image:
       torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
         --master_addr=MASTER_IP --master_port=29600 nccl_probe.py
   On node 1 use --node_rank=1. Use the real routable IP and cluster NIC
   configuration. Require all 16 ranks PASS.
5. Full official checkpoint: first 37,41 and 4K eager; capture API logits,
   then 39,39 with shared indices. Compare against a trusted run of the
   same checkpoint when available; PP-to-PP self-consistency alone
   cannot prove equivalence to the official W8A8 implementation.
   Try 32K/128K with MAX_MODEL_LEN >= prompt+32 and exact measured needle
   budgets, e.g. --needle 32000. Use MAX_NUM_SEQS=2 or more and
   --interleaved for genuine overlapping requests. Repeat with different
   MAX_BATCHED_TOKENS to test chunk boundaries. The probe checks chosen
   tokens/logprobs, not full-vocabulary equality or a quality benchmark.
   Preserve logs, JSON outputs, image ID, SIF hash and checkpoint revision.

Do not enable graphs, prefix caching, DFly2 or 1M as part of these initial
acceptance gates. Those require separate parity and capacity tests.

## Outstanding validation

No Docker build, GPU kernel execution or full-checkpoint inference has been
performed by this audit session. CPU sink reference: 7 tests passed locally; see AUDIT.md for scope. Source anchors checked against pinned
GitHub files. The source pin is an unmerged community PR; GPU regressions
are possible. Test K-cache insertion/gather, FP8 Marlin repacking and MoE,
iHC, gated MLA, actual model loading, PP transfer, prompt/response equivalence,
tool calls, then long context. No claim of A100 serving success or speed.
See ../PORTING_PLAN.md for external evidence and validation gates (older
CUDA 12.9 assumption superseded by this CUDA 13 plan).

CUDA 13.2.1 requires a compatible host driver; R595 is the matching branch. R580+ provides CUDA 13.x minor compatibility with limitations (in particular PTX/JIT features); do not assume JIT success just from nvidia-smi's CUDA version. See https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html.
