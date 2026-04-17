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

    # run everything
    ./run_wheel_test.py test all
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("CYLLAMA_MODELS_DIR", ROOT / "models"))

BACKENDS = {
    "cpu":    "cyllama",
    "cuda":   "cyllama-cuda12",
    "vulkan": "cyllama-vulkan",
    "rocm":   "cyllama-rocm",
    "sycl":   "cyllama-sycl",
}

# Env vars to set when running a given backend.
BACKEND_ENV = {
    "vulkan": {"GGML_VK_VISIBLE_DEVICES": "1"},
}


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


# Best-effort defaults — can be overridden via CYLLAMA_MODEL_<key>=repo_id:file
# or by placing files in MODELS_DIR yourself.
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
SD_REQUIREMENTS = ["z-image-turbo", "ae", "qwen3-4b"]
GEN_REQUIREMENTS = {
    "1": ["llama-3.2-1b"],
    "2": ["qwen3-4b"],
    "3": ["gemma-e4b"],
}


def _apply_env_overrides():
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

def run(cmd: list[str], env: dict | None = None, check: bool = True) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    result = subprocess.run(cmd, cwd=ROOT, env=full_env)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result.returncode


def cyllama(argv: list[str], env: dict | None = None) -> int:
    return run([sys.executable, "-m", "cyllama", *argv], env=env)


def cyllama_module(module: str, argv: list[str], env: dict | None = None) -> int:
    return run([sys.executable, "-m", module, *argv], env=env)


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


def env_for(backend: str) -> dict:
    return BACKEND_ENV.get(backend, {})


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

def _download_urllib(url: str, dest: Path):
    print(f"downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dest)


def _download_hf(repo_id: str, filename: str, dest: Path):
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print(
            "error: huggingface_hub not installed. "
            "Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"downloading {repo_id}:{filename} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=repo_id, filename=filename)
    shutil.copyfile(cached, dest)


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
        print(
            f"error: no source configured for model '{key}' ({src.filename}). "
            f"{src.notes}",
            file=sys.stderr,
        )
        sys.exit(2)
    return dest


def ensure_models(keys: list[str]) -> dict[str, Path]:
    return {k: ensure_model(k) for k in keys}


# ---------------------------------------------------------------------------
# tests (inlined from the shell scripts in ~/projects/demo/scripts)
# ---------------------------------------------------------------------------

def test_sd_1(backend: str):
    """z_turbo basic."""
    paths = ensure_models(SD_REQUIREMENTS)
    cyllama_module("cyllama.sd", [
        "txt2img",
        "--diffusion-model", str(paths["z-image-turbo"]),
        "--vae",             str(paths["ae"]),
        "--llm",             str(paths["qwen3-4b"]),
        "-H", "1024", "-W", "512",
        "-p", "a lovely cat",
    ], env=env_for(backend))


def test_sd_2(backend: str):
    """z_turbo cpu-offload."""
    paths = ensure_models(SD_REQUIREMENTS)
    cyllama_module("cyllama.sd", [
        "txt2img",
        "--diffusion-model", str(paths["z-image-turbo"]),
        "--vae",             str(paths["ae"]),
        "--llm",             str(paths["qwen3-4b"]),
        "--offload-to-cpu",
        "--vae-on-cpu",
        "-H", "1024", "-W", "512",
        "-p", "a lovely cat",
    ], env=env_for(backend))


def test_sd_3(backend: str):
    """z_turbo cpu-offload + flash-attn."""
    paths = ensure_models(SD_REQUIREMENTS)
    cyllama_module("cyllama.sd", [
        "txt2img",
        "--diffusion-model", str(paths["z-image-turbo"]),
        "--vae",             str(paths["ae"]),
        "--llm",             str(paths["qwen3-4b"]),
        "--cfg-scale", "1.0", "-v",
        "--offload-to-cpu",
        "--diffusion-fa",
        "-H", "1024", "-W", "512",
        "-p", "a lovely plump blue-eyed cat",
    ], env=env_for(backend))


def test_gen_1(backend: str):
    """Llama-3.2-1B short prompt."""
    model = ensure_model("llama-3.2-1b")
    cyllama(
        ["gen", "-m", str(model),
         "-p", "Explain quantum entanglement in one paragraph.",
         "-n", "256", "--stats"],
        env=env_for(backend),
    )


def test_gen_2(backend: str):
    """Qwen3-4B streamed."""
    model = ensure_model("qwen3-4b")
    cyllama(
        ["gen", "-m", str(model),
         "-p", "Write a haiku about GPUs.",
         "-n", "256", "--stream", "--stats"],
        env=env_for(backend),
    )


def test_gen_3(backend: str):
    """Gemma-4-E4B streamed."""
    model = ensure_model("gemma-e4b")
    cyllama(
        ["gen", "-m", str(model),
         "-p", "List three interesting facts about octopuses.",
         "-n", "512", "--temperature", "0.7", "--stream", "--stats"],
        env=env_for(backend),
    )


SD_TESTS = {"1": test_sd_1, "2": test_sd_2, "3": test_sd_3}
GEN_TESTS = {"1": test_gen_1, "2": test_gen_2, "3": test_gen_3}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_info(_args):
    backend = detect_backend()
    print(f"python:  {sys.executable}")
    print(f"backend: {backend or '(none)'}")
    print(f"models:  {MODELS_DIR}")
    if backend:
        cyllama(["info"])


def cmd_install(args):
    dist = BACKENDS[args.backend]
    cmd = [sys.executable, "-m", "pip", "install"]
    if args.upgrade:
        cmd.append("--upgrade")
    cmd.append(dist)
    run(cmd)


def cmd_download(args):
    if args.key == "all":
        keys = list(MODELS)
    else:
        keys = [args.key]
    for k in keys:
        try:
            path = ensure_model(k)
            print(f"ok: {k} -> {path}")
        except SystemExit:
            # ensure_model already printed an error; continue with others
            pass


def cmd_test(args):
    backend = require_backend(args.backend)
    mapping = SD_TESTS if args.kind == "sd" else GEN_TESTS
    keys = sorted(mapping) if args.n == "all" else [args.n]
    for k in keys:
        print(f"\n=== {args.kind} test {k} (backend={backend}) ===")
        mapping[k](backend)


def cmd_test_all(args):
    backend = require_backend(args.backend)
    for kind, mapping in (("sd", SD_TESTS), ("gen", GEN_TESTS)):
        for k in sorted(mapping):
            print(f"\n=== {kind} test {k} (backend={backend}) ===")
            mapping[k](backend)


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

    test = sub.add_parser("test", help="run a test")
    test.add_argument("kind", choices=["sd", "gen"])
    test.add_argument("n", choices=["1", "2", "3", "all"])
    test.add_argument("--backend", choices=list(BACKENDS))
    test.set_defaults(func=cmd_test)

    all_ = sub.add_parser("test-all", help="run every sd + gen test")
    all_.add_argument("--backend", choices=list(BACKENDS))
    all_.set_defaults(func=cmd_test_all)

    return p


def main():
    _apply_env_overrides()
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
