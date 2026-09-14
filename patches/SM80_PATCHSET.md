# DeepSeek-V4.1 SM80 patch lineage

Pinned source: `wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36`.

Important lineage:

- `64228995d75ae34b2d521bf38a9c4bfb962b2d2e` - DeepSeek-V4 SM8x Triton
  sparse-MLA foundation, software FP8 helper and pre-Hopper CUDA gates.
- `a59fd7919c446d178d53e69ffcfb22e691682009` - integrates upstream
  DeepSeek-V4.1 model support into the backport line.
- `4ba175b649874e4f37c99b0fe23b7a9fbc67476b` - V4.1 SM8x sparse-MLA
  backend (`TRITON_MLA_SPARSE_DSV41`) and Ampere routing.
- `1b84755041bd65f8e87c871eb35934ae7370ff64` - V4.1 FP8 cache/Engram
  software encode/decode for CUDA devices below SM89.
- `e21c36a912d2c4c4fe432bc04d92f9ff53aeaf76` - JIT warmup fixes for
  pre-SM89 CUDA.
- `e3377a16cddce6d3d92b6e44efe6917df00ff9ab` - V4.1-specific 128-token
  indexer backend outside SM90.
- `fae03f7118c9bc5c1a74e18ae859a9621bdfabb1` - DSpark V4.1 shard/local
  argmax hooks.

The SIF recipe does not replace the official image with the backport image.
It vendors only the Python/Triton layer listed in `OVERLAY_MANIFEST.md` while
keeping the official image's CUDA, PyTorch, vLLM native extensions and system
libraries.
