# DeepSeek-V4.1-Flash SM80 patch set

This repository intentionally pins a known working `wtdcode/vllm-backport`
revision/image instead of copying a large, fast-moving vLLM diff.

## Core V4.1/Ampere changes

| Commit | Purpose |
|---|---|
| `a59fd7919c446d178d53e69ffcfb22e691682009` | Merge upstream DeepSeek-V4.1 model support into the v0.13 port |
| `4ba175b649874e4f37c99b0fe23b7a9fbc67476b` | SM8x Triton sparse-MLA backend for DeepSeek-V4.1 |
| `1b84755041bd65f8e87c871eb35934ae7370ff64` | Software E4M3 encode/decode for FP8 cache paths on pre-SM89 GPUs |
| `fae03f7118c9bc5c1a74e18ae859a9621bdfabb1` | DSpark local-argmax / Markov hooks for V4.1 |
| `fbb35eea1044bdc542bb97d62edd7ea3a3ac51ec` | Accept `--tokenizer-mode deepseek_v41` |
| `e21c36a912d2c4c4fe432bc04d92f9ff53aeaf76` | Make JIT warmup compile below SM89 |
| `e3377a16cddce6d3d92b6e44efe6917df00ff9ab` | Give V4.1 its own indexer backend / 128-token pages |

Reproducibility pin:

```text
24cb31bb4fd0becee65c810c913a8caa4f610c36
```

Matching release image:

```text
lazymio/vllm-backport:v0.13.0-sm80
```

## Why not cherry-pick these onto arbitrary upstream vLLM?

V4.1 support crosses model integration, tokenizer/parser code, Engram, sparse
MLA, indexer cache layout, FP8 cache conversion, DSpark, and warmup. Upstream
was changing at the same time. A blind sequence of cherry-picks can build yet
still be wrong at runtime. Use the pinned fork revision or pinned SM80 image
first, then rebase only after the baseline serves correctly.
