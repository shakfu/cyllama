# PyPI Deployment

How cyllama and its GPU variants are published to PyPI, how to enable trusted publishing from CI, and how to harden the publish jobs with a GitHub environment.

## What gets published

As of v0.3.0 the published wheels are **abi3-only** (`cp312-abi3`, Python 3.12+). See [abi3.md](abi3.md) for the rationale.

cyllama is distributed as several **separate PyPI projects** — the base package plus one per GPU backend (the GPU workflows rename the package at build time via `CIBW_BEFORE_BUILD`):

| Project | Built by |
|---------|----------|
| `cyllama` (CPU) | `build-cibw-abi3.yml` |
| `cyllama-cuda12`, `cyllama-rocm`, `cyllama-sycl`, `cyllama-vulkan` | `build-gpu-wheels-abi3.yml` |

Each project has its own 10 GB PyPI size quota, which is the main reason for the abi3 switch (one wheel per platform instead of five).

## Two ways to publish

### 1. CI workflows, by trusted publishing

Both wheel workflows have a `publish_to_pypi` job. It uploads with [trusted publishing](https://docs.pypi.org/trusted-publishers/): GitHub issues the job an OIDC token, and PyPI exchanges it for an upload token valid for 15 minutes. No PyPI credential is stored in GitHub.

The job runs only when all of these hold:

- the repository variable `PYPI_TRUSTED_PUBLISHING` is `true` (the off switch; unset means off);
- the run is a tag build;
- the `publish_to_pypi` dispatch input is ticked;
- the build and smoke-test jobs passed.

The GPU workflow publishes with one job per distribution (a matrix over the four names). PyPI documents the minted token as project-scoped and the publish action as one upload per job, so each job mints its own token and uploads only its own wheels. A backend that was not built in the run is skipped.

To cut a release: push the tag, then run each workflow via **Actions -> (workflow) -> Run workflow** on the tag, with `publish_to_pypi` (and, if desired, `upload_release` for the GitHub Release) ticked.

### 2. Local manual publish

`make publish` runs `twine upload dist/*.whl` against whatever is in `dist/`, with the PyPI credentials configured on your machine. `make check` (a prerequisite of `publish`) refuses any non-abi3 wheel, so a stray per-version build cannot be uploaded by mistake. Because a single machine can only build one platform's wheels, the local path is mainly for quick single-platform fixes.

## Enabling trusted publishing

Do the PyPI side first. Until it is done, a publish job would fail authentication.

1. For each of the 5 projects, add a trusted publisher on PyPI: the project's **Settings** -> **Publishing** -> **Add a new publisher** -> **GitHub**. For a project that does not exist yet, add a *pending* publisher from your account's **Publishing** page; it creates the project on first upload, but does not reserve the name until then.

   | PyPI project | Workflow filename |
   |-|-|
   | `cyllama` | `build-cibw-abi3.yml` |
   | `cyllama-cuda12`, `cyllama-rocm`, `cyllama-sycl`, `cyllama-vulkan` | `build-gpu-wheels-abi3.yml` |

   For all five: owner `shakfu`, repository `cyllama`, environment `pypi`. The workflow must be the top-level file named above; trusted publishing does not work from a reusable workflow.

2. Create the `pypi` environment in GitHub and add required reviewers (next section).

3. Turn the switch on:

   ```sh
   gh variable set PYPI_TRUSTED_PUBLISHING --body true
   ```

   To turn it off again: `gh variable delete PYPI_TRUSTED_PUBLISHING`.

## Hardening: the `pypi` environment

Each `publish_to_pypi` job runs in the GitHub Actions **`pypi` environment** (`environment: pypi`), which the trusted publishers above also name. This lets you gate every PyPI upload behind environment protection rules:

1. Repository -> **Settings** -> **Environments** -> **New environment** -> name it `pypi`.

2. Add **Required reviewers** so a human must approve before any publish job runs. Optionally restrict deployments to tags via a deployment rule.

If the environment does not exist yet, GitHub creates it on first use with no protection rules, so configure the reviewers before turning on the switch.

## TestPyPI dry runs

The workflows publish to real PyPI (no `repository-url` set). To rehearse, register the same trusted publishers on TestPyPI and add `repository-url: https://test.pypi.org/legacy/` to the publish step.

Locally, `make publish-test` targets TestPyPI (`twine upload --repository testpypi`).
