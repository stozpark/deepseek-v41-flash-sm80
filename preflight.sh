#!/usr/bin/env bash
set -euo pipefail

need() { command -v "$1" >/dev/null 2>&1 || { echo "ERROR: missing command: $1" >&2; exit 1; }; }
need nvidia-smi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')
RAM_KB=$(awk '/MemTotal:/ {print $2}' /proc/meminfo)
RAM_GIB=$(( RAM_KB / 1024 / 1024 ))

echo "GPU count : $GPU_COUNT"
echo "Driver    : $DRIVER"
echo "Host RAM  : ${RAM_GIB} GiB"
echo
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if (( GPU_COUNT < 8 )); then
  echo "WARN: default recipe targets 8 GPUs (TP8). Found $GPU_COUNT." >&2
fi

BAD_GPU=0
while IFS= read -r name; do
  if [[ "$name" != *"A100"* && "$name" != *"A800"* ]]; then
    BAD_GPU=1
  fi
done < <(nvidia-smi --query-gpu=name --format=csv,noheader)
if (( BAD_GPU )); then
  echo "WARN: this branch targets SM80 A100/A800; another GPU was detected." >&2
fi

if (( RAM_GIB < 220 )); then
  echo "WARN: V4.1 Engram CPU offload is about 196 GB; host RAM is likely insufficient." >&2
elif (( RAM_GIB < 256 )); then
  echo "WARN: host RAM is very tight; 256 GiB+ is recommended." >&2
elif (( RAM_GIB < 384 )); then
  echo "NOTE: 384 GiB+ gives safer load/runtime headroom." >&2
fi

python3 - "$DRIVER" <<'PY'
import sys
v = sys.argv[1]
try:
    major = int(v.split('.')[0])
except Exception:
    major = 0
if major and major < 525:
    print("ERROR: CUDA 12.x minor-version compatibility requires NVIDIA driver >= 525.", file=sys.stderr)
    raise SystemExit(2)
if major and 525 <= major < 580:
    print("OK: driver is in NVIDIA's CUDA 12.x minor-version compatibility range (>=525,<580).")
    print("NOTE: features requiring a newer driver or unsupported PTX can still require a driver upgrade.")
elif major >= 580:
    print("OK: newer driver supports CUDA 12.x through backward compatibility.")
PY

echo
printf '%s\n' "Preflight finished."
