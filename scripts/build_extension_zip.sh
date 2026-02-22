#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

OUT_PATH="${1:-}"
if [[ -z "${OUT_PATH}" ]]; then
  MANIFEST_VERSION="$(node -p "JSON.parse(require('fs').readFileSync('manifest.json','utf8')).version")"
  OUT_PATH="dist/ai-content-screener-v${MANIFEST_VERSION}.zip"
fi

OUT_DIR="$(dirname "${OUT_PATH}")"
mkdir -p "${OUT_DIR}"

inputs=("manifest.json" "src")
if [[ -d "icon" ]]; then
  inputs+=("icon")
fi

zip -rq "${OUT_PATH}" "${inputs[@]}" -x "*.DS_Store"
echo "${OUT_PATH}"
