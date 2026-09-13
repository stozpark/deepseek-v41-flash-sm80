#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-18005}"
MODEL="${SERVED_MODEL_NAME:-deepseek-v4.1-flash}"
BASE="http://${HOST}:${PORT}"

echo "== /v1/models =="
curl -fsS "${BASE}/v1/models" | python3 -m json.tool

echo
echo "== chat smoke test =="
curl -fsS "${BASE}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: A100 OK\"}],\"temperature\":0,\"max_tokens\":32}" \
  | python3 -m json.tool
