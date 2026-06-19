#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
llama_dir="${repo_root}/thirdparty/llama.cpp"

if [[ ! -d "${llama_dir}/.git" && ! -f "${llama_dir}/.git" ]]; then
  echo "error: ${llama_dir} is not a git checkout" >&2
  exit 1
fi

latest_tag="$(
  git -C "${llama_dir}" for-each-ref refs/tags/b* \
    --sort=-creatordate \
    --format='%(refname:short)' \
    --count=1
)"

if [[ -z "${latest_tag}" ]]; then
  echo "error: no llama.cpp tags found; run update_llamacpp_tags.sh first" >&2
  exit 1
fi

pinned_commit="$(git -C "${llama_dir}" rev-parse HEAD)"
latest_tag_commit="$(git -C "${llama_dir}" rev-parse "${latest_tag}^{}")"

echo "Latest llama.cpp tag: ${latest_tag}"
echo "Latest tag commit:   ${latest_tag_commit}"
echo "Pinned commit:       ${pinned_commit}"
echo

if [[ "${latest_tag_commit}" == "${pinned_commit}" ]]; then
  echo "llama.cpp is already pinned to the latest tag."
  exit 0
fi

echo "llama.cpp is not pinned to the latest tag."
exit 2
