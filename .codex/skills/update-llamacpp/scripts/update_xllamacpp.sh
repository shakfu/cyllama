#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../../../.." && pwd)"
llama_dir="${repo_root}/thirdparty/llama.cpp"

if [[ -n "$(git -C "${repo_root}" status --porcelain --untracked-files=no)" ]]; then
  echo "error: xllamacpp has tracked local changes; refusing to update the parent repository" >&2
  git -C "${repo_root}" status --short --untracked-files=no >&2
  exit 1
fi

repo_default_branch="$(git -C "${repo_root}" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || true)"
repo_default_branch="${repo_default_branch#origin/}"
if [[ -z "${repo_default_branch}" ]]; then
  repo_default_branch="main"
fi

echo "Updating xllamacpp from origin/${repo_default_branch}..."
git -C "${repo_root}" fetch origin "${repo_default_branch}"
if git -C "${repo_root}" show-ref --verify --quiet "refs/heads/${repo_default_branch}"; then
  git -C "${repo_root}" switch "${repo_default_branch}"
else
  git -C "${repo_root}" switch --track -c "${repo_default_branch}" "origin/${repo_default_branch}"
fi
git -C "${repo_root}" pull --ff-only origin "${repo_default_branch}"

if [[ ! -d "${llama_dir}/.git" && ! -f "${llama_dir}/.git" ]]; then
  echo "error: ${llama_dir} is not a git checkout" >&2
  exit 1
fi

if [[ -n "$(git -C "${llama_dir}" status --porcelain)" ]]; then
  echo "error: thirdparty/llama.cpp has local changes; refusing to update" >&2
  git -C "${llama_dir}" status --short >&2
  exit 1
fi

default_branch="$(git -C "${llama_dir}" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null || true)"
default_branch="${default_branch#origin/}"
if [[ -z "${default_branch}" ]]; then
  default_branch="master"
fi

echo "Updating thirdparty/llama.cpp from origin/${default_branch}..."
git -C "${llama_dir}" fetch origin "${default_branch}"
if git -C "${llama_dir}" show-ref --verify --quiet "refs/heads/${default_branch}"; then
  git -C "${llama_dir}" switch "${default_branch}"
else
  git -C "${llama_dir}" switch --track -c "${default_branch}" "origin/${default_branch}"
fi
git -C "${llama_dir}" pull --ff-only origin "${default_branch}"

echo "Synchronizing llama.cpp submodules..."
git -C "${llama_dir}" submodule sync --recursive
git -C "${llama_dir}" submodule update --init --recursive

echo
echo "llama.cpp is now at:"
git -C "${llama_dir}" log -1 --oneline
echo
echo "Parent repository status:"
git -C "${repo_root}" status --short -- thirdparty/llama.cpp
