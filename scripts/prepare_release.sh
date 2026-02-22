#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  scripts/prepare_release.sh <version>

Examples:
  scripts/prepare_release.sh 0.1.1
  scripts/prepare_release.sh v0.1.1
EOF
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

INPUT_VERSION="$1"
VERSION="${INPUT_VERSION#v}"

if [[ ! "${VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Error: version must be semver format (e.g. 0.1.1)"
  exit 1
fi

TAG="v${VERSION}"
CURRENT_VERSION="$(node -p "JSON.parse(require('fs').readFileSync('manifest.json','utf8')).version")"

if [[ "${CURRENT_VERSION}" != "${VERSION}" ]]; then
  node -e "const fs=require('fs'); const p='manifest.json'; const m=JSON.parse(fs.readFileSync(p,'utf8')); m.version='${VERSION}'; fs.writeFileSync(p, JSON.stringify(m,null,2)+'\n');"
  echo "Updated manifest version: ${CURRENT_VERSION} -> ${VERSION}"
else
  echo "Manifest version is already ${VERSION}"
fi

CHECKED_VERSION="$(node -p "JSON.parse(require('fs').readFileSync('manifest.json','utf8')).version")"
if [[ "${CHECKED_VERSION}" != "${VERSION}" ]]; then
  echo "Error: manifest version check failed."
  exit 1
fi

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
  echo "Error: tag ${TAG} already exists locally."
  exit 1
fi

ZIP_PATH="dist/ai-content-screener-${TAG}.zip"
scripts/build_extension_zip.sh "${ZIP_PATH}" >/dev/null

echo "Prepared release artifacts:"
echo "- manifest version: ${VERSION}"
echo "- release tag: ${TAG}"
echo "- package: ${ZIP_PATH}"
echo
echo "Next commands:"
echo "  git add manifest.json"
echo "  git commit -m \"chore: release ${TAG}\""
echo "  git push origin develop"
echo "  # after merge to main:"
echo "  git checkout main && git pull origin main"
echo "  git tag ${TAG} && git push origin ${TAG}"
