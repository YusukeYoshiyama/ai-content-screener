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

MANIFEST_VERSION="$(node -p "JSON.parse(require('fs').readFileSync('manifest.json','utf8')).manifest_version")"
if [[ "${MANIFEST_VERSION}" != "3" ]]; then
  echo "Error: manifest_version must be 3 (current: ${MANIFEST_VERSION})"
  exit 1
fi

REQUIRED_ICON_PATHS=()
while IFS= read -r icon_path; do
  REQUIRED_ICON_PATHS+=("${icon_path}")
done < <(node - <<'NODE'
const fs = require("fs");
const manifest = JSON.parse(fs.readFileSync("manifest.json", "utf8"));
for (const path of Object.values(manifest.icons || {})) {
  if (path) {
    console.log(path);
  }
}
NODE
)

for icon_path in "${REQUIRED_ICON_PATHS[@]}"; do
  if [[ ! -f "${icon_path}" ]]; then
    echo "Error: required icon file is missing: ${icon_path}"
    exit 1
  fi
done

inputs=("manifest.json" "src")
if [[ -d "icon" ]]; then
  inputs+=("icon")
fi

zip -rq "${OUT_PATH}" "${inputs[@]}" -x "*.DS_Store"

for required_path in manifest.json "${REQUIRED_ICON_PATHS[@]}"; do
  if ! unzip -l "${OUT_PATH}" | awk '{print $4}' | grep -Fxq "${required_path}"; then
    echo "Error: required file is missing in zip: ${required_path}"
    exit 1
  fi
done

echo "${OUT_PATH}"
