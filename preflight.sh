#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${ROOT_DIR}/VERSION.env"

command -v nvidia-smi >/dev/null 2>&1 || { echo "ERROR: nvidia-smi not found"; exit 1; }
mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
DRIVER_MAJOR="${DRIVER%%.*}"
GPU_COUNT="${#GPU_NAMES[@]}"
MEM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
MEM_GIB="$((MEM_KB / 1024 / 1024))"

echo "official_image=${OFFICIAL_IMAGE}"
echo "cuda_family=${CUDA_FAMILY}"
echo "driver=${DRIVER}"
echo "gpus=${GPU_COUNT}"
printf '  %s\n' "${GPU_NAMES[@]}"
echo "host_ram_gib=${MEM_GIB}"

for name in "${GPU_NAMES[@]}"; do
  [[ "$name" == *A100* || "$name" == *A800* ]] || echo "WARN: non-A100/A800 GPU detected: $name"
done
(( GPU_COUNT >= 8 )) || echo "WARN: default launcher expects 8 GPUs (TP8)."
(( MEM_GIB >= 220 )) || echo "WARN: host RAM is likely insufficient for Engram CPU offload."
(( MEM_GIB >= 384 )) || echo "WARN: 384 GiB+ host RAM is preferred; 256 GiB is tight."

case "$CUDA_FAMILY" in
  13.*)
    (( DRIVER_MAJOR >= 580 )) || echo "WARN: CUDA 13 branch is safest on R580+; use cu129 branch on older data-center drivers."
    ;;
  12.9*)
    (( DRIVER_MAJOR >= 525 )) || echo "WARN: CUDA 12.x requires an R525+ data-center driver family."
    ;;
esac
