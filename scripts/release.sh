#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  scripts/release.sh build-zip [out-path]
  scripts/release.sh prepare <version>

Examples:
  scripts/release.sh build-zip
  scripts/release.sh build-zip dist/ai-content-screener-v0.1.1.zip
  scripts/release.sh prepare 0.1.1
  scripts/release.sh prepare v0.1.1
EOF
}

manifest_value() {
  local key="$1"
  node -p "JSON.parse(require('fs').readFileSync('manifest.json','utf8'))['${key}']"
}

required_icon_paths() {
  node - <<'NODE'
const fs = require("fs");
const manifest = JSON.parse(fs.readFileSync("manifest.json", "utf8"));
for (const path of Object.values(manifest.icons || {})) {
  if (path) {
    console.log(path);
  }
}
NODE
}

build_zip() {
  local out_path="${1:-}"
  if [[ -z "${out_path}" ]]; then
    local manifest_version
    manifest_version="$(manifest_value version)"
    out_path="dist/ai-content-screener-v${manifest_version}.zip"
  fi

  local out_dir
  out_dir="$(dirname "${out_path}")"
  mkdir -p "${out_dir}"

  local manifest_version
  manifest_version="$(manifest_value manifest_version)"
  if [[ "${manifest_version}" != "3" ]]; then
    echo "Error: manifest_version must be 3 (current: ${manifest_version})"
    exit 1
  fi

  local icon_path
  while IFS= read -r icon_path; do
    if [[ -n "${icon_path}" && ! -f "${icon_path}" ]]; then
      echo "Error: required icon file is missing: ${icon_path}"
      exit 1
    fi
  done < <(required_icon_paths)

  local inputs=("manifest.json" "src")
  if [[ -d "icon" ]]; then
    inputs+=("icon")
  fi

  zip -rq "${out_path}" "${inputs[@]}" -x "*.DS_Store"

  local required_path
  while IFS= read -r required_path; do
    if [[ -z "${required_path}" ]]; then
      continue
    fi
    if ! unzip -l "${out_path}" | awk '{print $4}' | grep -Fxq "${required_path}"; then
      echo "Error: required file is missing in zip: ${required_path}"
      exit 1
    fi
  done < <(printf '%s\n' manifest.json && required_icon_paths)

  echo "${out_path}"
}

prepare_release() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 1
  fi

  local input_version="$1"
  local version="${input_version#v}"
  if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: version must be semver format (e.g. 0.1.1)"
    exit 1
  fi

  local tag="v${version}"
  local current_version
  current_version="$(manifest_value version)"

  if [[ "${current_version}" != "${version}" ]]; then
    node -e "const fs=require('fs'); const p='manifest.json'; const m=JSON.parse(fs.readFileSync(p,'utf8')); m.version='${version}'; fs.writeFileSync(p, JSON.stringify(m,null,2)+'\n');"
    echo "Updated manifest version: ${current_version} -> ${version}"
  else
    echo "Manifest version is already ${version}"
  fi

  local checked_version
  checked_version="$(manifest_value version)"
  if [[ "${checked_version}" != "${version}" ]]; then
    echo "Error: manifest version check failed."
    exit 1
  fi

  if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
    echo "Error: tag ${tag} already exists locally."
    exit 1
  fi

  local zip_path
  zip_path="dist/ai-content-screener-${tag}.zip"
  build_zip "${zip_path}" >/dev/null

  echo "Prepared release artifacts:"
  echo "- manifest version: ${version}"
  echo "- release tag: ${tag}"
  echo "- package: ${zip_path}"
  echo
  echo "Next commands:"
  echo "  git add manifest.json"
  echo "  git commit -m \"chore: release ${tag}\""
  echo "  git push origin <branch>"
  echo "  # after merge to main:"
  echo "  git checkout main && git pull origin main"
  echo "  git tag ${tag} && git push origin ${tag}"
}

main() {
  local command="${1:-}"
  case "${command}" in
    build-zip)
      shift
      if [[ $# -gt 1 ]]; then
        usage
        exit 1
      fi
      build_zip "${1:-}"
      ;;
    prepare)
      shift
      prepare_release "$@"
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

main "$@"
