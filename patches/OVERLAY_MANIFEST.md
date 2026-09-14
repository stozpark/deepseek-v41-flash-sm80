# SM80 vendored overlay

The A100/SM80 runtime sources are committed under `patches/vendor/`; SIF build does not clone GitHub, fetch a commit, use curl, or download patch files.

Source pin: `wtdcode/vllm-backport@24cb31bb4fd0becee65c810c913a8caa4f610c36`.
`patches/vendor/BACKPORT_COMMIT` records the pin and `patches/vendor/SHA256SUMS` verifies the vendored source snapshot before build.
