#!/usr/bin/env bash
set -euo pipefail
cat >&2 <<'MSG'
This repository intentionally patches the official DeepSeek-V4.1 image while
building a SIF. The unmodified official Docker tag is not the A100 runtime.
Use:
  ./build_sif.sh
  ./serve_tp8_a100.sh
MSG
exit 2
