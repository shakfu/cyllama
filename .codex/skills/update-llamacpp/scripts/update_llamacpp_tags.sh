#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
llama_dir="${repo_root}/thirdparty/llama.cpp"

if [[ ! -d "${llama_dir}/.git" && ! -f "${llama_dir}/.git" ]]; then
  echo "error: ${llama_dir} is not a git checkout" >&2
  exit 1
fi

remote_url="$(git -C "${llama_dir}" remote get-url origin)"
echo "Refreshing llama.cpp tags from origin:"
echo "  ${remote_url}"
echo

before_count="$(git -C "${llama_dir}" tag --list | wc -l | tr -d '[:space:]')"
before_latest="$(git -C "${llama_dir}" tag --list --sort=-creatordate | sed -n '1p' || true)"

git -C "${llama_dir}" fetch origin --tags --force --prune --prune-tags

after_count="$(git -C "${llama_dir}" tag --list | wc -l | tr -d '[:space:]')"
after_latest="$(git -C "${llama_dir}" tag --list --sort=-creatordate | sed -n '1p' || true)"

echo
echo "Tag count: ${before_count} -> ${after_count}"
if [[ -n "${before_latest}" || -n "${after_latest}" ]]; then
  echo "Latest tag by creator date: ${before_latest:-<none>} -> ${after_latest:-<none>}"
fi
echo
echo "Most recent tags:"
git -C "${llama_dir}" tag --list --sort=-creatordate | sed -n '1,10p'
