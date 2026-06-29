# PyPI Deployment

How cyllama and its GPU variants are published to PyPI, how to configure the publishing token, and how to harden the publish jobs with a GitHub environment.

## What gets published

As of v0.3.0 the published wheels are **abi3-only** (`cp312-abi3`, Python 3.12+). See [abi3.md](abi3.md) for the rationale.

cyllama is distributed as several **separate PyPI projects** — the base package plus one per GPU backend (the GPU workflows rename the package at build time via `CIBW_BEFORE_BUILD`):

| Project | Built by |
|---------|----------|
| `cyllama` (CPU) | `build-cibw-abi3.yml` (and `build-cibw.yml` for per-version builds) |
| `cyllama-cuda12`, `cyllama-rocm`, `cyllama-sycl`, `cyllama-vulkan` | `build-gpu-wheels-abi3.yml` (and `build-gpu-wheels.yml`) |

Each project has its own 10 GB PyPI size quota, which is the main reason for the abi3 switch (one wheel per platform instead of five).

## Two ways to publish

### 1. CI workflows (recommended for releases)

Each of the four wheel workflows has an **optional** `publish_to_pypi` job, gated behind a `workflow_dispatch` input (default `false`). It is inert on normal CI runs; publishing only happens when you explicitly enable it at dispatch time.

The job downloads the collected-wheels artifact and uploads via `pypa/gh-action-pypi-publish` using an API token:

```yaml
- name: Publish to PyPI
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    packages-dir: dist
    password: ${{ secrets.PYPI_API_TOKEN }}
    skip-existing: true
```

`skip-existing: true` makes a re-run after a partial multi-wheel upload idempotent (PyPI forbids overwriting an existing version regardless).

To cut a release: trigger the workflow via **Actions -> (workflow) -> Run workflow**, set `publish_to_pypi` (and, if desired, `upload_release` for the GitHub Release) to `true`.

### 2. Local manual publish

`make publish` runs `twine upload dist/*.whl` against whatever is in `dist/`. `make check` (a prerequisite of `publish`) refuses any non-abi3 wheel, so a stray per-version build cannot be uploaded by mistake. Because a single machine can only build one platform's wheels, the local path is mainly for quick single-platform fixes; full releases come from CI.

## Creating the PyPI API token

Authentication is via an **API token** (consistently — we do not use OIDC trusted publishing; see the note at the end for why).

1. Log in at <https://pypi.org> -> **Account settings** -> **API tokens** -> **Add API token**.

2. Name it, e.g. `cyllama-github-actions`.

3. **Scope** — choose **"Entire account"** (account-wide). This is required because the GPU workflows upload several distinct projects (`cyllama-cuda12`, `cyllama-vulkan`, ...) in a single step, so one account-scoped token must cover them all. Project-scoped tokens are more restrictive but would need one token per project and could not drive the GPU single-step upload from one secret.

4. Copy the token — it is shown **once** and looks like `pypi-AgEIcHlwaS5vcmc...`.

A project-scoped token can only be created after the project already exists on PyPI. An account-scoped token avoids this chicken-and-egg for new projects.

## Adding the token as a GitHub secret

The secret name **must be exactly `PYPI_API_TOKEN`** — that is what all four workflows reference.

- **Web UI**: repository -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**. Name `PYPI_API_TOKEN`, value = the full `pypi-...` string.

- **gh CLI** (prompts for the value, so it stays out of shell history):

  ```sh
  gh secret set PYPI_API_TOKEN
  ```

You do **not** set a username: `pypa/gh-action-pypi-publish` uses `__token__` automatically when the token is passed as `password:`.

## Hardening: the `pypi` environment

Each `publish_to_pypi` job runs in the GitHub Actions **`pypi` environment** (`environment: pypi`). This lets you gate every PyPI upload behind environment protection rules:

1. Repository -> **Settings** -> **Environments** -> **New environment** -> name it `pypi`.

2. Add **Required reviewers** so a human must approve before any publish job runs. Optionally restrict to specific branches/tags via a deployment branch rule.

3. (Optional, more restrictive) Move the `PYPI_API_TOKEN` secret from a repository secret to an **environment secret** scoped to `pypi`, so the token is only readable by jobs that target that environment.

The environment is referenced by name in the workflows; if it does not exist yet, GitHub creates it on first use with no protection rules, so configure the reviewers before relying on the gate.

## TestPyPI dry runs

The workflows publish to real PyPI (no `repository-url` set). To rehearse against TestPyPI instead, add a separate TestPyPI token and point the action at the test index:

```yaml
- uses: pypa/gh-action-pypi-publish@release/v1
  with:
    packages-dir: dist
    password: ${{ secrets.TEST_PYPI_API_TOKEN }}
    repository-url: https://test.pypi.org/legacy/
    skip-existing: true
```

Locally, `make publish-test` already targets TestPyPI (`twine upload --repository testpypi`).

## Why token auth, not OIDC

PyPI's OIDC trusted publishing converts cleanly for the single-project CPU workflows, but not for the GPU workflows: OIDC mint tokens are project-scoped, so the one mixed-`dist/` upload of four distinct projects would have to be split into per-project upload steps, each with its own trusted-publisher registration (per project x per workflow filename). A single account-scoped API token uploads everything in one step, so we use tokens consistently across all four workflows.
