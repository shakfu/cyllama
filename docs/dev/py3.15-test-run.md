# Python 3.15 test run (2026-10-10)

The first local run of cyllama on CPython 3.15.0 final. Machine: M1 MacBook Air, 16 GB, Metal backend, llama.cpp `b11429`.

## Result

On 3.15.0, cyllama builds from source and passes the suite. The only failures are pre-existing `tests/test_decision.py` failures. They reproduce identically on 3.13.2.

| Run | Result |
|-|-|
| 3.15.0, `--deselect tests/test_decision.py` | 2506 passed, 231 skipped, 0 failed |
| 3.15.0, full suite, before the M1 skip | 88 failed, 2350 passed, 98 errors |
| 3.13.2 (`make test`), full suite, before the M1 skip | 98 failed, 2353 passed, 98 errors |
| 3.13.2 (`make test`), full suite, with the M1 skip | 3 failed (laya drift), 2542 passed, 237 skipped |
| 3.15.0, full suite, with the M1 skip | 3 failed (laya drift), 2529 passed, 238 skipped |

The pass-count gap is mostly langchain tests, which skip on 3.15 (see below).

## Setup

- `.venv-315` is a separate venv; `.venv`, `uv.lock` and `make test` are untouched. Build output goes to `build/cp315-cp315-macosx_26_0_arm64/`.
- uv 0.12.24 (Homebrew, 2026-10-08) only offers `3.15.0rc3`. The final build is in python-build-standalone release `20261009`, which this uv does not know. The interpreter was fetched from that release, checked against its `SHA256SUMS`, and passed to `uv venv --python`. Once a newer uv ships, use `uv python install 3.15` instead.
- Install: `uv pip install -p .venv-315 -e . pytest pytest-asyncio pytest-cov pytest-memray pytest-mock numpy requests`.
- Run: `.venv-315/bin/python -m pytest`.

This tested an editable cp315 build, not the published cp312-abi3 wheel loaded on 3.15. `pyproject.toml` classifiers stop at 3.14.

## Findings

### langchain does not install on 3.15

`langchain` -> `langgraph` -> `langgraph-checkpoint` -> `ormsgpack` 1.12.2. It has no cp315 wheel. Its source build uses PyO3 0.27.2, which rejects 3.15 (max 3.14). langchain was left out of the venv. Its integration tests are `importorskip`-guarded, so they skipped and did not run.

### `asyncio.iscoroutinefunction` is deprecated

`src/cyllama/agents/workflow.py:400` and `:1960` emit `DeprecationWarning` (284 warnings in `test_agents_workflow.py`). Deprecated in 3.14, removal in 3.16 ([gh-122875](https://github.com/python/cpython/issues/122875)). Replace with `inspect.iscoroutinefunction`, available on all supported versions. It differs only for objects carrying asyncio's private `_is_coroutine` marker. No other warning was new to 3.15.

### `test_decision.py` failures cascade across the suite

Two separate failures, both on 3.13 and 3.15:

1. `lev` and `kev` (4.2 GB Q8_0) fail with `kIOGPUCommandBufferCallbackErrorOutOfMemory` (`llama_process failed with code -3`). After that, every later model load in the same process fails ("Failed to load model from file"). This causes all of the ~190 failures and errors from `test_generate.py` onward. GPU in-use memory was 66 MB and system memory 86% free before the run, so the cause is not leftover memory pressure.
2. `laya` `angry` drifts outside `TOL = 2e-3`. It fails `TestModels::test_matches_reference[laya]` and `test_server_systemone[embedded|python]`. Every other laya probability is within tolerance.

| Source | `angry` |
|-|-|
| Reference (`REFERENCE["laya"]`, llama-server b11429, CPU) | 0.788043 |
| M1, Metal (`n_gpu_layers=-1`) | 0.785367 (diff 0.0027) |
| M1, CPU (`n_gpu_layers=0`) | 0.783817 (diff 0.0042) |

The CPU result is further off than Metal, so this is not a GPU-tolerance issue. The llama.cpp pin matches the reference. The M1 reports `hw.optional.arm.FEAT_I8MM: 0`. M2 and later have I8MM, and ggml uses different Q8_0 kernels when it is present. Inference, unconfirmed: the reference was captured on an I8MM machine, and the summation order differs. To confirm, run the CPU check on the M5 Pro.

## Change

`tests/test_decision.py` now skips `lev` and `kev` when `machdep.cpu.brand_string` starts with `Apple M1`. That prefix also matches M1 Pro/Max/Ultra, including 32-64 GB machines that may not OOM. Test lev/kev on the M5 Pro. The `laya` drift is left failing; it is not a memory issue.

## Revisit on a larger machine (open)

Run on the M5 Pro (or any non-M1 machine with more RAM) and record results here:

1. `uv run pytest tests/test_decision.py -m slow` with `lev` and `kev` present. Do they pass, and is GPU memory sufficient?
2. Laya CPU check: `DecisionModel('models/Laya-Q8_0.gguf', n_gpu_layers=0).answer(README_REQUEST)["answers"]["angry"]`.
   - 0.788043 (within 2e-3): the reference depends on the hardware. Decide: per-machine reference, or skip on M1.
   - Otherwise: investigate the code. Do not widen `TOL`.
3. Same check with `n_gpu_layers=-1` (Metal).
4. Decide the skip condition: keep `Apple M1` prefix, or gate on memory (`hw.memsize` or Metal `recommendedMaxWorkingSetSize`).
5. Still open from the 3.15 run: replace `asyncio.iscoroutinefunction` in `src/cyllama/agents/workflow.py`; retry langchain on 3.15 once `ormsgpack` ships cp315 wheels; test the cp312-abi3 wheel on 3.15.
