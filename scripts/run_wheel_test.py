#!/usr/bin/env python3
"""Self-contained smoke-test runner for built cyllama wheels.

Supports cpu / cuda / vulkan / rocm / sycl backends, can download the
required models from the Hugging Face Hub, and runs stable-diffusion and
text-generation tests as inline Python functions (no shell scripts
required).

Examples:
    # install a wheel into the current environment
    ./run_wheel_test.py install vulkan

    # download everything this script needs
    ./run_wheel_test.py download all

    # run a single test, backend auto-detected from installed distribution
    ./run_wheel_test.py test gen 1

    # run every sd + gen test (continues on failure, prints summary)
    ./run_wheel_test.py test all all

    # stop at the first failing test
    ./run_wheel_test.py test all all --fail-fast

    # per-invocation timeout
    ./run_wheel_test.py test sd 1 --timeout 600
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("CYLLAMA_MODELS_DIR", ROOT / "models"))

BACKENDS: dict[str, str] = {
    "cpu":    "cyllama",
    "cuda":   "cyllama-cuda12",
    "vulkan": "cyllama-vulkan",
    "rocm":   "cyllama-rocm",
    "sycl":   "cyllama-sycl",
}

# Default env for a given backend. Existing values in os.environ take
# precedence -- only unset keys are populated from these defaults, so
# callers can always override by exporting the variable themselves.
BACKEND_ENV_DEFAULTS: dict[str, dict[str, str]] = {
    # Pin Vulkan to a specific device by default; override with
    # GGML_VK_VISIBLE_DEVICES=... in the caller's env if needed.
    "vulkan": {"GGML_VK_VISIBLE_DEVICES": "1"},
}


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------

class ModelSourceUnavailable(RuntimeError):
    """Raised when a model has no configured source and isn't on disk."""


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelSource:
    """Where to fetch a model from.

    One of repo_id (HF Hub) or url (direct http) must be set.
    """

    filename: str
    repo_id: str | None = None
    hf_filename: str | None = None  # defaults to filename
    url: str | None = None
    notes: str = ""

    def hub_filename(self) -> str:
        return self.hf_filename or self.filename


# Best-effort defaults -- can be overridden via CYLLAMA_MODEL_<KEY>=repo_id:file
# or by placing files in MODELS_DIR yourself. Use `list-models` to inspect.
MODELS: dict[str, ModelSource] = {
    "llama-3.2-1b": ModelSource(
        filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    ),
    "qwen3-4b": ModelSource(
        filename="Qwen3-4B-Q8_0.gguf",
        repo_id="Qwen/Qwen3-4B-GGUF",
    ),
    "gemma-e4b": ModelSource(
        filename="gemma-4-E4B-it-Q5_K_M.gguf",
        repo_id="",  # override via env if/when available
        notes="set CYLLAMA_MODEL_GEMMA_E4B=<repo_id>:<hf_filename> to enable download",
    ),
    "z-image-turbo": ModelSource(
        filename="z_image_turbo-Q6_K.gguf",
        repo_id="",
        notes="set CYLLAMA_MODEL_Z_IMAGE_TURBO=<repo_id>:<hf_filename> to enable download",
    ),
    "ae": ModelSource(
        filename="ae.safetensors",
        repo_id="black-forest-labs/FLUX.1-schnell",
        hf_filename="ae.safetensors",
    ),
}

# Which tests need which models.
SD_REQUIREMENTS: list[str] = ["z-image-turbo", "ae", "qwen3-4b"]


def _apply_env_overrides() -> None:
    """Allow overriding repo ids via env vars (CYLLAMA_MODEL_<KEY>=repo:file)."""
    for key, src in MODELS.items():
        env_key = "CYLLAMA_MODEL_" + key.upper().replace("-", "_")
        val = os.environ.get(env_key)
        if not val:
            continue
        if ":" in val:
            repo, fname = val.split(":", 1)
            src.repo_id = repo
            src.hf_filename = fname
        else:
            src.repo_id = val


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------

def run(
    cmd: list[str],
    env: dict[str, str] | None = None,
    check: bool = False,
    timeout: float | None = None,
) -> int:
    """Run a subprocess; return the exit code.

    Unlike previous revisions, `check=False` is the default so callers
    can accumulate failures across a smoke-test matrix. Pass
    ``check=True`` to restore the old fail-fast behaviour.
    """
    print(f"$ {' '.join(cmd)}", flush=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        result = subprocess.run(cmd, cwd=ROOT, env=full_env, timeout=timeout)
        rc = result.returncode
    except subprocess.TimeoutExpired:
        print(f"error: command timed out after {timeout}s", file=sys.stderr)
        rc = 124  # conventional timeout exit code
    if check and rc != 0:
        sys.exit(rc)
    return rc


def cyllama(argv: list[str], env: dict[str, str] | None = None, timeout: float | None = None) -> int:
    return run([sys.executable, "-m", "cyllama", *argv], env=env, timeout=timeout)


def cyllama_module(
    module: str,
    argv: list[str],
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> int:
    return run([sys.executable, "-m", module, *argv], env=env, timeout=timeout)


# ---------------------------------------------------------------------------
# backend detection / install
# ---------------------------------------------------------------------------

def detect_backend() -> str | None:
    for backend, dist in BACKENDS.items():
        try:
            md.distribution(dist)
            return backend
        except md.PackageNotFoundError:
            continue
    return None


def env_for(backend: str) -> dict[str, str]:
    """Return default env overrides for a backend, skipping keys the
    caller has already set in the surrounding environment."""
    defaults = BACKEND_ENV_DEFAULTS.get(backend, {})
    return {k: v for k, v in defaults.items() if k not in os.environ}


def require_backend(requested: str | None) -> str:
    detected = detect_backend()
    if requested and detected and requested != detected:
        print(
            f"warning: requested backend '{requested}' but '{detected}' is installed",
            file=sys.stderr,
        )
    backend = requested or detected
    if not backend:
        print(
            "error: no cyllama backend installed. Run: "
            f"{Path(__file__).name} install {{{','.join(BACKENDS)}}}",
            file=sys.stderr,
        )
        sys.exit(2)
    return backend


# ---------------------------------------------------------------------------
# model download
# ---------------------------------------------------------------------------

def _download_urllib(url: str, dest: Path) -> None:
    print(f"downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_report = time.monotonic()
    bytes_read = 0
    chunk = 1024 * 1024  # 1 MiB
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total_hdr = r.headers.get("Content-Length")
        total = int(total_hdr) if total_hdr and total_hdr.isdigit() else None
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            bytes_read += len(buf)
            now = time.monotonic()
            if now - last_report >= 2.0:
                if total:
                    pct = 100.0 * bytes_read / total
                    print(
                        f"  {bytes_read / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.1f}%)",
                        flush=True,
                    )
                else:
                    print(f"  {bytes_read / 1e6:.1f} MB", flush=True)
                last_report = now
    tmp.rename(dest)


def _download_hf(repo_id: str, filename: str, dest: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "error: huggingface_hub not installed. "
            "Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"downloading {repo_id}:{filename} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Land the file directly in MODELS_DIR rather than copying from the
    # HF cache. Newer huggingface_hub uses `local_dir_use_symlinks=False`
    # and places the file at `<local_dir>/<filename>`; older releases
    # fall back to the cache path which we then copy.
    try:
        out = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(dest.parent),
            local_dir_use_symlinks=False,
        )
    except TypeError:
        # Older huggingface_hub without local_dir kwarg.
        out = hf_hub_download(repo_id=repo_id, filename=filename)
    out_path = Path(out)
    if out_path != dest:
        shutil.copyfile(out_path, dest)


def ensure_model(key: str) -> Path:
    src = MODELS[key]
    dest = MODELS_DIR / src.filename
    if dest.exists():
        return dest
    if src.url:
        _download_urllib(src.url, dest)
    elif src.repo_id:
        _download_hf(src.repo_id, src.hub_filename(), dest)
    else:
        raise ModelSourceUnavailable(
            f"no source configured for model '{key}' ({src.filename}). {src.notes}"
        )
    return dest


def ensure_models(keys: list[str]) -> dict[str, Path]:
    return {k: ensure_model(k) for k in keys}


# ---------------------------------------------------------------------------
# tests (inlined from the shell scripts in ~/projects/demo/scripts)
# ---------------------------------------------------------------------------

def test_sd_1(backend: str, timeout: float | None) -> int:
    """z_turbo basic."""
    paths = ensure_models(SD_REQUIREMENTS)
    return cyllama_module("cyllama.sd", [
        "txt2img",
        "--diffusion-model", str(paths["z-image-turbo"]),
        "--vae",             str(paths["ae"]),
        "--llm",             str(paths["qwen3-4b"]),
        "-H", "1024", "-W", "512",
        "-p", "a lovely cat",
    ], env=env_for(backend), timeout=timeout)


def test_sd_2(backend: str, timeout: float | None) -> int:
    """z_turbo cpu-offload."""
    paths = ensure_models(SD_REQUIREMENTS)
    return cyllama_module("cyllama.sd", [
        "txt2img",
        "--diffusion-model", str(paths["z-image-turbo"]),
        "--vae",             str(paths["ae"]),
        "--llm",             str(paths["qwen3-4b"]),
        "--offload-to-cpu",
        "--vae-on-cpu",
        "-H", "1024", "-W", "512",
        "-p", "a lovely cat",
    ], env=env_for(backend), timeout=timeout)


def test_sd_3(backend: str, timeout: float | None) -> int:
    """z_turbo cpu-offload + flash-attn."""
    paths = ensure_models(SD_REQUIREMENTS)
    return cyllama_module("cyllama.sd", [
        "txt2img",
        "--diffusion-model", str(paths["z-image-turbo"]),
        "--vae",             str(paths["ae"]),
        "--llm",             str(paths["qwen3-4b"]),
        "--cfg-scale", "1.0", "-v",
        "--offload-to-cpu",
        "--diffusion-fa",
        "-H", "1024", "-W", "512",
        "-p", "a lovely plump blue-eyed cat",
    ], env=env_for(backend), timeout=timeout)


def test_gen_1(backend: str, timeout: float | None) -> int:
    """Llama-3.2-1B short prompt."""
    model = ensure_model("llama-3.2-1b")
    return cyllama(
        ["gen", "-m", str(model),
         "-p", "Explain quantum entanglement in one paragraph.",
         "-n", "256", "--stats"],
        env=env_for(backend), timeout=timeout,
    )


def test_gen_2(backend: str, timeout: float | None) -> int:
    """Qwen3-4B streamed."""
    model = ensure_model("qwen3-4b")
    return cyllama(
        ["gen", "-m", str(model),
         "-p", "Write a haiku about GPUs.",
         "-n", "256", "--stream", "--stats"],
        env=env_for(backend), timeout=timeout,
    )


def test_gen_3(backend: str, timeout: float | None) -> int:
    """Gemma-4-E4B streamed."""
    model = ensure_model("gemma-e4b")
    return cyllama(
        ["gen", "-m", str(model),
         "-p", "List three interesting facts about octopuses.",
         "-n", "512", "--temperature", "0.7", "--stream", "--stats"],
        env=env_for(backend), timeout=timeout,
    )


SD_TESTS = {"1": test_sd_1, "2": test_sd_2, "3": test_sd_3}
GEN_TESTS = {"1": test_gen_1, "2": test_gen_2, "3": test_gen_3}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_info(_args: argparse.Namespace) -> int:
    backend = detect_backend()
    print(f"{'python:':<9}{sys.executable}")
    print(f"{'backend:':<9}{backend or '(none)'}")
    print(f"{'models:':<9}{MODELS_DIR}")
    if backend:
        cyllama(["info"])
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    dist = BACKENDS[args.backend]
    cmd = [sys.executable, "-m", "pip", "install"]
    if args.upgrade:
        cmd.append("--upgrade")
    cmd.append(dist)
    return run(cmd)


def cmd_download(args: argparse.Namespace) -> int:
    keys = list(MODELS) if args.key == "all" else [args.key]
    failures = 0
    for k in keys:
        try:
            path = ensure_model(k)
            print(f"ok: {k} -> {path}")
        except ModelSourceUnavailable as e:
            print(f"skip: {k}: {e}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def cmd_list_models(_args: argparse.Namespace) -> int:
    for key, src in MODELS.items():
        source = (
            f"hf:{src.repo_id}:{src.hub_filename()}"
            if src.repo_id
            else (src.url or "(no source configured)")
        )
        on_disk = "YES" if (MODELS_DIR / src.filename).exists() else "no"
        print(f"{key:<16} file={src.filename:<40} on_disk={on_disk:<3} source={source}")
        if src.notes and not src.repo_id and not src.url:
            print(f"{'':<16} note: {src.notes}")
    return 0


def cmd_list_tests(_args: argparse.Namespace) -> int:
    for kind, mapping in (("sd", SD_TESTS), ("gen", GEN_TESTS)):
        for n, fn in sorted(mapping.items()):
            doc = (fn.__doc__ or "").strip()
            print(f"{kind} {n}  {doc}")
    return 0


def _collect_runs(kind: str, n: str) -> list[tuple[str, str]]:
    """Expand ('all'|'sd'|'gen', 'all'|'1'|...) into concrete (kind, n) pairs."""
    kinds = ["sd", "gen"] if kind == "all" else [kind]
    runs: list[tuple[str, str]] = []
    for k in kinds:
        mapping = SD_TESTS if k == "sd" else GEN_TESTS
        keys = sorted(mapping) if n == "all" else [n]
        for nk in keys:
            runs.append((k, nk))
    return runs


def cmd_test(args: argparse.Namespace) -> int:
    backend = require_backend(args.backend)
    runs = _collect_runs(args.kind, args.n)
    if args.dry_run:
        for k, n in runs:
            print(f"would run: {k} {n} (backend={backend})")
        return 0

    results: list[tuple[str, str, int]] = []
    for k, n in runs:
        mapping = SD_TESTS if k == "sd" else GEN_TESTS
        print(f"\n=== {k} test {n} (backend={backend}) ===")
        try:
            rc = mapping[n](backend, args.timeout)
        except ModelSourceUnavailable as e:
            print(f"skip: {e}", file=sys.stderr)
            rc = 2
        results.append((k, n, rc))
        if rc != 0 and args.fail_fast:
            break

    # Summary
    print("\n=== summary ===")
    worst = 0
    for k, n, rc in results:
        status = "PASS" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  {k} {n}: {status}")
        worst = max(worst, rc)
    passed = sum(1 for _, _, rc in results if rc == 0)
    print(f"{passed}/{len(results)} passed")
    return worst


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="cyllama wheel tester")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="show python/backend/models info").set_defaults(func=cmd_info)

    inst = sub.add_parser("install", help="pip install a cyllama backend wheel")
    inst.add_argument("backend", choices=list(BACKENDS))
    inst.add_argument("--upgrade", action="store_true")
    inst.set_defaults(func=cmd_install)

    dl = sub.add_parser("download", help="download a model (or 'all')")
    dl.add_argument("key", choices=[*MODELS.keys(), "all"])
    dl.set_defaults(func=cmd_download)

    sub.add_parser(
        "list-models",
        help="list known model keys, their filenames and sources",
    ).set_defaults(func=cmd_list_models)

    sub.add_parser(
        "list-tests",
        help="list available smoke tests",
    ).set_defaults(func=cmd_list_tests)

    test = sub.add_parser("test", help="run one or more smoke tests")
    test.add_argument("kind", choices=["sd", "gen", "all"])
    test.add_argument("n", choices=["1", "2", "3", "all"])
    test.add_argument("--backend", choices=list(BACKENDS))
    test.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-test timeout in seconds (default: no timeout)",
    )
    test.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failing test instead of running the full matrix",
    )
    test.add_argument(
        "--dry-run",
        action="store_true",
        help="print the test matrix without downloading or invoking anything",
    )
    test.set_defaults(func=cmd_test)

    return p


def main() -> None:
    _apply_env_overrides()
    args = build_parser().parse_args()
    rc = args.func(args)
    sys.exit(int(rc or 0))


if __name__ == "__main__":
    main()
