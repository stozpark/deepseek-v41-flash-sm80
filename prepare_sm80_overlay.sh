#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${ROOT_DIR}/patches/generated"
PATCH="${PATCH_DIR}/sm80-cu129-minimal.patch"
SHA="${PATCH_DIR}/sm80-cu129-minimal.patch.sha256"
MANIFEST="${PATCH_DIR}/sm80-cu129-minimal.manifest.txt"

for f in "$PATCH" "$SHA" "$MANIFEST" "${ROOT_DIR}/patches/apply_minimal_patch.py"; do
  [[ -f "$f" ]] || { echo "ERROR: missing production SM80 patch input: $f" >&2; exit 1; }
done

(
  cd "$PATCH_DIR"
  sha256sum -c "$(basename "$SHA")"
)
[[ "$(sed -n 's/^changed_files=//p' "$MANIFEST")" == "25" ]] || {
  echo "ERROR: production minimal patch must contain exactly 25 changed files" >&2
  exit 1
}
if grep -Eq '^models/deepseek_v4/(amd|cpu|xpu)/|^models/deepseek_v4_1/amd/' "$MANIFEST"; then
  echo "ERROR: non-A100 backend present in production patch manifest" >&2
  exit 1
fi

echo "[ok] verified production-minimal SM80 patch"
echo "[ok] patch=${PATCH}"
echo "[info] patches/vendor is retained only for audit/regeneration; production SIF does not inject it"
