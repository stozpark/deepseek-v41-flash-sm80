#!/usr/bin/env bash
# Serve two independent TP8 replicas on ONE 16xA100 host (not TP16).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found." >&2
    exit 2
  fi
  available="$(nvidia-smi -L | grep -c -E 'GPU [0-9]+:')" || true
  if (( available < 16 )); then
    echo "ERROR: expected 16 local GPUs, found ${available}." >&2
    echo "For two 8xGPU nodes, run serve_a100.sh separately on each host." >&2
    exit 2
  fi
fi
PORT_A="${PORT_A:-18005}"
PORT_B="${PORT_B:-18006}"
if [[ "${PORT_A}" == "${PORT_B}" ]]; then
  echo "ERROR: replica ports must differ." >&2
  exit 2
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  echo "Replica 1:"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT="${PORT_A}" \
    SERVED_MODEL_NAME=deepseek-v41-flash-1017-a \
    DRY_RUN=1 TP_SIZE=8 bash "${ROOT}/serve_a100.sh"
  echo "Replica 2:"
  CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 PORT="${PORT_B}" \
    SERVED_MODEL_NAME=deepseek-v41-flash-1017-b \
    DRY_RUN=1 TP_SIZE=8 bash "${ROOT}/serve_a100.sh"
  exit 0
fi
echo "Launching 2 independent TP8 replicas; each loads its own full model."
echo "WARNING: Engram host RAM is shared; verify RAM/pinned memory capacity."
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT="${PORT_A}" \
  SERVED_MODEL_NAME=deepseek-v41-flash-1017-a \
  TP_SIZE=8 bash "${ROOT}/serve_a100.sh" &
PID_A=$!
CUDA_VISIBLE_DEVICES=8,9,10,11,12,13,14,15 PORT="${PORT_B}" \
  SERVED_MODEL_NAME=deepseek-v41-flash-1017-b \
  TP_SIZE=8 bash "${ROOT}/serve_a100.sh" &
PID_B=$!
cleanup() {
  kill "${PID_A}" "${PID_B}" 2>/dev/null || true
  wait "${PID_A}" "${PID_B}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
wait -n "${PID_A}" "${PID_B}" || {
  status=$?
  echo "One TP8 replica failed (exit ${status}); stopping the other." >&2
  exit "${status}"
}
echo "A TP8 replica exited; stopping both." >&2
