#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${ROOT_DIR}/prepare_sm80_overlay.sh"
echo "Production source patch is verified and ready for ./build_sif.sh"
