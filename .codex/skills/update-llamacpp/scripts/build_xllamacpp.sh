#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${TMPDIR:-/tmp}/xllamacpp-build-${timestamp}.log"

if [[ ! -f "${repo_root}/scripts/build.py" ]]; then
  echo "error: missing ${repo_root}/scripts/build.py" >&2
  exit 1
fi

if [[ "$(uname -s)" == "Darwin" && -z "${MACOSX_DEPLOYMENT_TARGET:-}" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=13.3
  echo "MACOSX_DEPLOYMENT_TARGET was unset; using ${MACOSX_DEPLOYMENT_TARGET}"
fi

echo "Building xllamacpp via scripts/build.py"
echo "Build log: ${log_file}"
echo

set +e
python3 "${repo_root}/scripts/build.py" 2>&1 | tee "${log_file}"
status="${PIPESTATUS[0]}"
set -e

echo
if [[ "${status}" -eq 0 ]]; then
  echo "xllamacpp build succeeded."
else
  echo "xllamacpp build failed with exit code ${status}."
  echo "Inspect log: ${log_file}"
fi

exit "${status}"
