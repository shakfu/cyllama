#!/usr/bin/env python3
"""Self-contained smoke-test runner for built cyllama wheels.

``--venv`` names the environment under test and every subprocess runs that
interpreter directly. Without it the script falls back to ``uv run``, which
re-syncs whichever project owns the cwd -- so from the cyllama checkout it
would build the extension from source and test that, never the wheel.

``--cuda`` (and ``--cpu`` / ``--vulkan`` / ``--rocm`` / ``--sycl``) is shorthand
for ``--venv .venv-<backend> --backend <backend>``; anything given explicitly
wins over it. ``--wheel`` installs a local wheel into that venv, creating it if
needed, and ``--pypi`` installs from PyPI instead; either runs implicitly
before a test target. Options are accepted before or after the target.

Each test is one target -- ``test-all``, ``test-gen-all``, ``test-sd-3`` --
named identically to the generated Makefile rules; ``list-tests`` prints them.

Examples:
    # create .venv-cuda, install a local wheel into it, run everything
    python run_wheel_test.py --cuda --wheel dist/cyllama_cuda12-0.4.3-cp312-abi3-win_amd64.whl test-all

    # against an existing venv; backend is detected from what is installed
    python run_wheel_test.py --venv .venv-cuda12 test-all

    # one family, or one case
    python run_wheel_test.py --cuda test-rag-all
    python run_wheel_test.py --cuda test-sd-3 --timeout 600

    # install from PyPI rather than a local wheel
    python run_wheel_test.py --vulkan test-all --pypi
    python run_wheel_test.py --venv .venv-x --pypi cyllama-cuda12==0.4.3 test-gen-1

    # show the matrix without installing, downloading or running anything
    python run_wheel_test.py --cuda test-all --dry-run

    # environment, registry and target listings
    python run_wheel_test.py --cuda info
    python run_wheel_test.py list-tests
    python run_wheel_test.py --models-dir models download all

``--pypi`` takes an optional spec, so a bare ``--pypi`` must not sit directly
before the target -- put it last, or give it a spec.
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


def _find_root() -> Path:
    """Locate the project root: the cwd for subprocesses and the parent of
    ``models/`` and ``.venv/``.

    This file is checked in as ``<repo>/scripts/run_wheel_test.py`` but is also
    meant to be copied out standalone (as ``./run.py``) into a bare
    uv-managed wheel-test directory. Walking up to the nearest project marker
    handles both layouts; using ``__file__``'s own directory would resolve to
    ``<repo>/scripts`` in-repo and download models to ``scripts/models``.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return here


ROOT = _find_root()
MODELS_DIR = Path(os.environ.get("CYLLAMA_MODELS_DIR", ROOT / "models"))

# Resolve `uv` once. Everything this script shells out to Python for is
# routed through `uv run` so it executes inside the project's uv venv
# regardless of how the script itself was launched.
UV = shutil.which("uv") or "uv"

# The environment under test. When --venv is given, every subprocess runs that
# interpreter *directly* rather than through `uv run`. This matters: `uv run`
# re-syncs whichever project owns the cwd, so run from this checkout it would
# build cyllama from source and test that instead of the installed wheel.
# Set by _apply_cli_overrides(); None restores the legacy `uv run` behaviour.
VENV: Path | None = None

# Interpreter `uv venv` should build the target env from (--python). Left unset,
# uv picks its own default, which is not necessarily the version a given wheel
# was built for.
VENV_PYTHON: str | None = None

# Where the --cpu/--cuda/--vulkan/... shorthands look for their venv. Relative
# names resolve against ROOT so the shorthand means the same thing from any cwd.
DEFAULT_VENV_PREFIX = os.environ.get("CYLLAMA_VENV_PREFIX", ".venv-")


def venv_python(venv: Path) -> Path:
    """Interpreter path inside `venv`, on either the Windows or POSIX layout."""
    win = venv / "Scripts" / "python.exe"
    if win.exists():
        return win
    posix = venv / "bin" / "python"
    if posix.exists():
        return posix
    return win if os.name == "nt" else posix


def ensure_venv(venv: Path) -> Path:
    """Create `venv` if it does not exist yet; return its interpreter."""
    py = venv_python(venv)
    if not py.exists():
        print(f"creating venv at {venv}")
        cmd = [UV, "venv", str(venv)]
        if VENV_PYTHON:
            cmd += ["--python", VENV_PYTHON]
        subprocess.run(cmd, check=True)
        py = venv_python(venv)
    return py


def python_cmd() -> list[str]:
    """argv prefix that runs Python in the environment under test."""
    if VENV is not None:
        return [str(venv_python(VENV))]
    return [UV, "run", "python"]


def pip_install(
    spec: list[str],
    upgrade: bool = False,
    reinstall: bool = False,
    extra: list[str] | None = None,
) -> int:
    """Install `spec` (plus any --with packages) into the environment under test."""
    cmd = [UV, "pip", "install"]
    if VENV is not None:
        cmd += ["--python", str(ensure_venv(VENV))]
    if upgrade:
        cmd.append("--upgrade")
    if reinstall:
        cmd.append("--reinstall")
    return run(cmd + spec + list(extra or []))


BACKENDS: dict[str, str] = {
    "cpu": "cyllama",
    "cuda": "cyllama-cuda12",
    "vulkan": "cyllama-vulkan",
    "rocm": "cyllama-rocm",
    "sycl": "cyllama-sycl",
}

# Default env for a given backend. Existing values in os.environ take
# precedence -- only unset keys are populated from these defaults, so
# callers can always override by exporting the variable themselves.
BACKEND_ENV_DEFAULTS: dict[str, dict[str, str]] = {
    # Every subprocess here goes through `uv run`, which re-syncs the project
    # environment first. Against an installed wheel that is a no-op, but in an
    # editable checkout it *rebuilds the extension* -- and the backend is chosen
    # from the environment at compile time, so without GGML_CUDA=1 the rebuild
    # links a CPU-only extension against CUDA static libs and every test dies
    # with `undefined symbol: ggml_backend_cuda_reg`. Set it so a dev checkout
    # rebuilds for the backend it is being asked to test.
    "cuda": {"GGML_CUDA": "1"},
    "rocm": {"GGML_HIP": "1"},
    "sycl": {"GGML_SYCL": "1"},
    # Same reasoning, plus: pin Vulkan to a specific device by default;
    # override with GGML_VK_VISIBLE_DEVICES=... in the caller's env if needed.
    "vulkan": {"GGML_VULKAN": "1", "GGML_VK_VISIBLE_DEVICES": "1"},
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
        url="https://huggingface.co/hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF/resolve/main/llama-3.2-1b-instruct-q8_0.gguf",
    ),
    "qwen3-4b": ModelSource(
        filename="Qwen3-4B-Q8_0.gguf",
        repo_id="Qwen/Qwen3-4B-GGUF",
        url="https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q8_0.gguf",
    ),
    "gemma-e4b": ModelSource(
        filename="gemma-4-E4B-it-Q5_K_M.gguf",
        repo_id="",  # override via env if/when available
        notes="set CYLLAMA_MODEL_GEMMA_E4B=<repo_id>:<hf_filename> to enable download",
        url="https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q5_K_M.gguf",
    ),
    "z-image-turbo": ModelSource(
        filename="z_image_turbo-Q6_K.gguf",
        repo_id="",
        notes="set CYLLAMA_MODEL_Z_IMAGE_TURBO=<repo_id>:<hf_filename> to enable download",
        url="https://huggingface.co/unsloth/Z-Image-Turbo-GGUF/resolve/main/z-image-turbo-Q6_K.gguf",
    ),
    "ae": ModelSource(
        filename="ae.safetensors",
        repo_id="black-forest-labs/FLUX.1-schnell",
        hf_filename="ae.safetensors",
        url="https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/vae/ae.safetensors",
    ),
    "bge-small-en": ModelSource(
        filename="bge-small-en-v1.5-q8_0.gguf",
        repo_id="CompendiumLabs/bge-small-en-v1.5-gguf",
        url="https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf/resolve/main/bge-small-en-v1.5-q8_0.gguf",
    ),
    "whisper-base-en": ModelSource(
        filename="ggml-base.en.bin",
        repo_id="ggerganov/whisper.cpp",
        url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
    ),
}

# Which tests need which models.
SD_REQUIREMENTS: list[str] = ["z-image-turbo", "ae", "qwen3-4b"]
RAG_REQUIREMENTS: list[str] = ["qwen3-4b", "bge-small-en"]

# ---------------------------------------------------------------------------
# data assets (corpus text, sample audio)
#
# These are inputs rather than models. In the checkout they already exist under
# tests/media; standalone they do not, so each has a fallback -- jfk.wav is
# fetched from whisper.cpp, and the corpus is synthesised rather than
# downloaded, since the one in the repo is a copyrighted short story.
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.environ.get("CYLLAMA_DATA_DIR", ROOT / "tests" / "media"))

# The checkout keeps text under tests/media but audio under tests/samples, so
# look in both rather than making the caller pick one with --data-dir.
DATA_FALLBACK_DIRS: list[Path] = [ROOT / "tests" / "media", ROOT / "tests" / "samples"]


def find_data_asset(name: str) -> Path | None:
    """First existing copy of `name` in --data-dir or the checkout's data dirs."""
    for d in (DATA_DIR, *DATA_FALLBACK_DIRS):
        candidate = d / name
        if candidate.exists():
            return candidate
    return None


JFK_WAV_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"

# One text per line -- the format `cyllama embed -f` expects. Deliberately
# includes a cluster about mortality so the `--similarity "death and dying"`
# query in the embed case has something to rank above its 0.5 threshold.
GENERATED_CORPUS: list[str] = [
    "The old man knew that he was dying, and he felt no fear of it.",
    "Death comes for everyone eventually, and grief is the price of having loved.",
    "Mourners gathered at the graveside in the cold morning air.",
    "He had spent his last years writing about mortality and the end of life.",
    "The hospice nurse spoke gently about what the final days would be like.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "The compiler performs constant folding before emitting machine code.",
    "Mount Kilimanjaro is the highest free-standing mountain in the world.",
    "She sold the bakery and moved to a small town near the coast.",
    "Quicksort has an average time complexity of O(n log n).",
    "The bridge was rebuilt after the flood washed away its central span.",
    "A leopard was found frozen near the western summit of the mountain.",
]


def ensure_corpus() -> Path:
    """Path to a line-per-text corpus, preferring the checkout's own."""
    repo_copy = find_data_asset("corpus1.txt")
    if repo_copy is not None:
        return repo_copy
    generated = MODELS_DIR / "corpus_generated.txt"
    if not generated.exists():
        print(f"writing generated corpus -> {generated}")
        generated.parent.mkdir(parents=True, exist_ok=True)
        generated.write_text("\n".join(GENERATED_CORPUS) + "\n", encoding="utf-8")
    return generated


def ensure_audio() -> Path:
    """Path to the jfk.wav sample, downloading it if the checkout lacks one."""
    repo_copy = find_data_asset("jfk.wav")
    if repo_copy is not None:
        return repo_copy
    dest = MODELS_DIR / "jfk.wav"
    if not dest.exists():
        _download_urllib(JFK_WAV_URL, dest)
    return dest


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


def _kill_tree(proc: "subprocess.Popen[bytes]") -> None:
    """Kill `proc` and every process it spawned.

    ``proc.kill()`` reaps only the direct child. A venv's python.exe re-execs
    the real interpreter, so a timed-out image run leaves that grandchild alive
    holding several GiB of VRAM -- and every later test in the matrix then OOMs
    or crawls, which silently invalidates the whole run's timings. Take the
    entire tree down instead.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        import signal

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        print(f"warning: could not fully reap pid {proc.pid}", file=sys.stderr)


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
    # Redirected stdout on Windows defaults to the ANSI codepage, and the sd log
    # callback emits byte-level BPE markers (U+0120, U+010A) that cp1252 cannot
    # encode -- one UnicodeEncodeError traceback per log line once the output is
    # piped to a file. Force UTF-8 so a logged run matches a console one.
    full_env.setdefault("PYTHONIOENCODING", "utf-8")
    if env:
        full_env.update(env)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, start_new_session=os.name != "nt")
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"error: command timed out after {timeout}s", file=sys.stderr)
        _kill_tree(proc)
        rc = 124  # conventional timeout exit code
    if check and rc != 0:
        sys.exit(rc)
    return rc


def cyllama(argv: list[str], env: dict[str, str] | None = None, timeout: float | None = None) -> int:
    return run([*python_cmd(), "-m", "cyllama", *argv], env=env, timeout=timeout)


def cyllama_module(
    module: str,
    argv: list[str],
    env: dict[str, str] | None = None,
    timeout: float | None = None,
) -> int:
    return run([*python_cmd(), "-m", module, *argv], env=env, timeout=timeout)


# ---------------------------------------------------------------------------
# backend detection / install
# ---------------------------------------------------------------------------


_DETECT_SRC = """
import importlib.metadata as md
for backend, dist in {backends!r}.items():
    try:
        md.distribution(dist)
        print(backend)
        break
    except md.PackageNotFoundError:
        pass
"""


def _detect_backend_in_venv(venv: Path) -> str | None:
    py = venv_python(venv)
    if not py.exists():
        return None
    proc = subprocess.run(
        [str(py), "-c", _DETECT_SRC.format(backends=BACKENDS)],
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip() or None


def detect_backend() -> str | None:
    # With an explicit target venv, ask *it* what is installed. importlib.metadata
    # here would describe the interpreter running this script, which under
    # `uv run` from the checkout is the project env, not the wheel under test.
    if VENV is not None:
        return _detect_backend_in_venv(VENV)
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
        name = Path(__file__).name
        if VENV is not None:
            print(
                f"error: no cyllama backend installed in {VENV}."
                f"\n  Install a local wheel:  {name} --venv {VENV} --wheel <path> test all all"
                f"\n  ...or from PyPI:        {name} --venv {VENV} --pypi --backend vulkan test all all",
                file=sys.stderr,
            )
        else:
            print(
                f"error: no cyllama backend installed. Run: {name} install {{{','.join(BACKENDS)}}}",
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
            "error: huggingface_hub not installed. Install with: pip install huggingface_hub",
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
        raise ModelSourceUnavailable(f"no source configured for model '{key}' ({src.filename}). {src.notes}")
    return dest


def ensure_models(keys: list[str]) -> dict[str, Path]:
    return {k: ensure_model(k) for k in keys}


# ---------------------------------------------------------------------------
# tests (inlined from the shell scripts in ~/projects/demo/scripts)
# ---------------------------------------------------------------------------


def test_sd_1(backend: str, timeout: float | None) -> int:
    """z_turbo te-on-cpu."""
    # Unqualified, this case is a pure-GPU run needing ~9.4 GiB (3.9 text
    # encoder + 5.5 diffusion) and OOMs on anything smaller: upstream
    # master-731 dropped `free_params_immediately`, so the conditioner's
    # weights now stay resident for the life of the context instead of being
    # freed once the prompt is encoded. Parking the text encoder's weights in
    # RAM frees enough for the diffusion model while every module still
    # computes on the GPU -- unlike test 2, which moves *all* the weights.
    #
    # --vae-tiling is not optional here. Placement alone still dies in VAE
    # decode: at 512x1024 it wants a 3328 MiB compute buffer with the 5.5 GiB
    # of diffusion weights still resident, and no `--params-backend` spelling
    # helps because that is a compute buffer, not weights (`te=cpu,vae=cpu`
    # fails identically). Tiling is what shrinks it.
    #
    # Measured on an 8 GiB RTX 4060: 3.17 s/it, 69 s end to end. `--auto-fit`
    # also fits but declines the GPU altogether on a single-GPU box (~143 s/it),
    # which no wheel-test timeout would survive.
    paths = ensure_models(SD_REQUIREMENTS)
    return cyllama_module(
        "cyllama.sd",
        [
            "txt2img",
            "--diffusion-model",
            str(paths["z-image-turbo"]),
            "--vae",
            str(paths["ae"]),
            "--llm",
            str(paths["qwen3-4b"]),
            "--params-backend",
            "te=cpu",
            "--vae-tiling",
            "-H",
            "1024",
            "-W",
            "512",
            "-o",
            "z_turbo_1.png",
            "-p",
            "a lovely cat",
        ],
        env=env_for(backend),
        timeout=timeout,
    )


def test_sd_2(backend: str, timeout: float | None) -> int:
    """z_turbo cpu-offload."""
    paths = ensure_models(SD_REQUIREMENTS)
    return cyllama_module(
        "cyllama.sd",
        [
            "txt2img",
            "--diffusion-model",
            str(paths["z-image-turbo"]),
            "--vae",
            str(paths["ae"]),
            "--llm",
            str(paths["qwen3-4b"]),
            "--offload-to-cpu",
            "--vae-on-cpu",
            "-H",
            "1024",
            "-W",
            "512",
            "-o",
            "z_turbo_2.png",
            "-p",
            "a lovely cat",
        ],
        env=env_for(backend),
        timeout=timeout,
    )


def test_sd_3(backend: str, timeout: float | None) -> int:
    """z_turbo cpu-offload + flash-attn."""
    paths = ensure_models(SD_REQUIREMENTS)
    return cyllama_module(
        "cyllama.sd",
        [
            "txt2img",
            "--diffusion-model",
            str(paths["z-image-turbo"]),
            "--vae",
            str(paths["ae"]),
            "--llm",
            str(paths["qwen3-4b"]),
            "--cfg-scale",
            "1.0",
            "-v",
            "--offload-to-cpu",
            "--diffusion-fa",
            "-H",
            "1024",
            "-W",
            "512",
            "-o",
            "z_turbo_3.png",
            "-p",
            "a lovely plump blue-eyed cat",
        ],
        env=env_for(backend),
        timeout=timeout,
    )


def test_gen_1(backend: str, timeout: float | None) -> int:
    """Llama-3.2-1B short prompt."""
    model = ensure_model("llama-3.2-1b")
    return cyllama(
        ["gen", "-m", str(model), "-p", "Explain quantum entanglement in one paragraph.", "-n", "256", "--stats"],
        env=env_for(backend),
        timeout=timeout,
    )


def test_gen_2(backend: str, timeout: float | None) -> int:
    """Qwen3-4B streamed."""
    model = ensure_model("qwen3-4b")
    return cyllama(
        ["gen", "-m", str(model), "-p", "Write a haiku about GPUs.", "-n", "256", "--stream", "--stats"],
        env=env_for(backend),
        timeout=timeout,
    )


def test_gen_3(backend: str, timeout: float | None) -> int:
    """Gemma-4-E4B streamed."""
    model = ensure_model("gemma-e4b")
    return cyllama(
        [
            "gen",
            "-m",
            str(model),
            "-p",
            "List three interesting facts about octopuses.",
            "-n",
            "512",
            "--temperature",
            "0.7",
            "--stream",
            "--stats",
        ],
        env=env_for(backend),
        timeout=timeout,
    )


def has_module(name: str) -> bool:
    """Whether `name` is importable in the environment under test."""
    proc = subprocess.run(
        [*python_cmd(), "-c", f"import {name}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def test_embed_1(backend: str, timeout: float | None) -> int:
    """corpus similarity ranking."""
    model = ensure_model("bge-small-en")
    corpus = ensure_corpus()
    return cyllama(
        [
            "embed",
            "-m",
            str(model),
            "-f",
            str(corpus),
            "--similarity",
            "death and dying",
            "--threshold",
            "0.5",
        ],
        env=env_for(backend),
        timeout=timeout,
    )


def test_transcribe_1(backend: str, timeout: float | None) -> int:
    """jfk.wav speech-to-text."""
    # The invariant is that transcription works on a bare wheel install: the
    # wheels declare no dependencies, so nothing on this path may import a
    # third-party package. Do not gate on numbers being present -- gating on
    # numpy would fail a *correctly* built wheel, which is the whole point.
    model = ensure_model("whisper-base-en")
    audio = ensure_audio()
    rc = cyllama(
        ["transcribe", "-f", str(audio), "-m", str(model)],
        env=env_for(backend),
        timeout=timeout,
    )
    if rc != 0 and not has_module("numpy"):
        # Wheels built before numpy was removed from whisper/cli.py import it at
        # module scope while declaring no dependency on it, so they die on the
        # import rather than on anything whisper did.
        print(
            "hint: this wheel may predate the numpy removal in whisper/cli.py."
            "\n  Re-running with --with numpy will confirm that diagnosis;"
            "\n  if it then passes, the wheel needs rebuilding, not a dependency.",
            file=sys.stderr,
        )
    return rc


def test_rag_1(backend: str, timeout: float | None) -> int:
    """in-memory index + query."""
    paths = ensure_models(RAG_REQUIREMENTS)
    corpus = ensure_corpus()
    return cyllama(
        [
            "rag",
            "-m",
            str(paths["qwen3-4b"]),
            "-e",
            str(paths["bge-small-en"]),
            "-f",
            str(corpus),
            # The case script omits -p and drops into an interactive chat loop,
            # which a smoke test cannot drive; a single query exercises the same
            # index -> retrieve -> generate path and then exits.
            "-p",
            "What does this text say about death?",
            "-n",
            "128",
            "--sources",
        ],
        env=env_for(backend),
        timeout=timeout,
    )


def test_rag_2(backend: str, timeout: float | None) -> int:
    """persistent sqlite vector store (build + reopen)."""
    paths = ensure_models(RAG_REQUIREMENTS)
    corpus = ensure_corpus()
    db = ROOT / "vector.db"
    if db.exists():
        db.unlink()  # start from nothing so the create path is covered

    def query(prompt: str) -> int:
        return cyllama(
            [
                "rag",
                "-m",
                str(paths["qwen3-4b"]),
                "-e",
                str(paths["bge-small-en"]),
                "-f",
                str(corpus),
                "--db",
                str(db),
                "-p",
                prompt,
                "-n",
                "128",
            ],
            env=env_for(backend),
            timeout=timeout,
        )

    rc = query("What does this text say about death?")
    if rc != 0:
        return rc
    if not db.exists():
        print(f"error: --db was given but no store was created at {db}", file=sys.stderr)
        return 1
    # Second pass reopens the existing store instead of re-embedding: the whole
    # point of --db, and the only part a single run would not cover.
    print(f"-- reopening existing store ({db.stat().st_size} bytes)")
    return query("What is the mountain in this text?")


SD_TESTS = {"1": test_sd_1, "2": test_sd_2, "3": test_sd_3}
GEN_TESTS = {"1": test_gen_1, "2": test_gen_2, "3": test_gen_3}
EMBED_TESTS = {"1": test_embed_1}
TRANSCRIBE_TESTS = {"1": test_transcribe_1}
RAG_TESTS = {"1": test_rag_1, "2": test_rag_2}

# Every test family, in the order `test all all` runs them: cheap and
# fast-failing first, the multi-minute image cases last.
TEST_FAMILIES: dict[str, dict[str, "Callable[[str, float | None], int]"]] = {
    "embed": EMBED_TESTS,
    "transcribe": TRANSCRIBE_TESTS,
    "gen": GEN_TESTS,
    "rag": RAG_TESTS,
    "sd": SD_TESTS,
}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_info(_args: argparse.Namespace) -> int:
    backend = detect_backend()
    target = str(venv_python(VENV)) if VENV is not None else sys.executable
    print(f"{'python:':<9}{target}")
    print(f"{'backend:':<9}{backend or '(none)'}")
    print(f"{'models:':<9}{MODELS_DIR}")
    if backend:
        cyllama(["info"])
    return 0


def cmd_sync(_args: argparse.Namespace) -> int:
    return run([UV, "sync"])


def cmd_clean(_args: argparse.Namespace) -> int:
    venv = VENV if VENV is not None else ROOT / ".venv"
    if venv.exists():
        print(f"removing {venv}")
        shutil.rmtree(venv)
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    rc = cmd_clean(args)
    if rc != 0:
        return rc
    return cmd_sync(args)


def resolve_install_spec(args: argparse.Namespace) -> list[str] | None:
    """Work out what to install, if anything: a local wheel or a PyPI spec.

    --wheel wins over --pypi. A bare --pypi needs --backend to know which
    distribution to ask for; --pypi cyllama-cuda12==0.4.3 names it outright.
    """
    wheel = getattr(args, "wheel", None)
    if wheel:
        path = Path(wheel).expanduser().resolve()
        if not path.exists():
            print(f"error: wheel not found: {path}", file=sys.stderr)
            sys.exit(2)
        return [str(path)]

    pypi = getattr(args, "pypi", None)
    if not pypi:
        return None
    if pypi is not True:  # an explicit requirement spec
        return [pypi]
    backend = getattr(args, "backend", None)
    if not backend:
        print(
            f"error: --pypi without a spec needs --backend {{{','.join(BACKENDS)}}} "
            f"(or give one, e.g. --pypi cyllama-vulkan==0.4.3)",
            file=sys.stderr,
        )
        sys.exit(2)
    return [BACKENDS[backend]]


def cmd_install(args: argparse.Namespace) -> int:
    spec = resolve_install_spec(args)
    if spec is None:
        # Neither --wheel nor --pypi: the positional backend means PyPI.
        if not getattr(args, "backend", None):
            print("error: give a backend, --wheel <path>, or --pypi [spec]", file=sys.stderr)
            return 2
        spec = [BACKENDS[getattr(args, "backend")]]
    return pip_install(
        spec,
        upgrade=getattr(args, "upgrade", False),
        reinstall=getattr(args, "reinstall", False),
        extra=getattr(args, "extra", None),
    )


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
        source = f"hf:{src.repo_id}:{src.hub_filename()}" if src.repo_id else (src.url or "(no source configured)")
        on_disk = "YES" if (MODELS_DIR / src.filename).exists() else "no"
        print(f"{key:<16} file={src.filename:<40} on_disk={on_disk:<3} source={source}")
        if src.notes and not src.repo_id and not src.url:
            print(f"{'':<16} note: {src.notes}")
    return 0


# Human-readable section headings for the generated Makefile's help text.
FAMILY_TITLES: dict[str, str] = {
    "embed": "Embedding",
    "transcribe": "Transcription",
    "gen": "Generation",
    "rag": "RAG",
    "sd": "Stable Diffusion",
}


def _render_makefile() -> str:
    py_var = "uv run ./run.py"
    backends = list(BACKENDS)

    family_targets: dict[str, list[str]] = {
        fam: [f"test-{fam}-{n}" for n in sorted(mapping)] + [f"test-{fam}-all"]
        for fam, mapping in TEST_FAMILIES.items()
    }
    width = max(len(t) for ts in family_targets.values() for t in ts) + 2

    # Group .PHONY into readable lines
    groups = [
        ["help", "sync", "info", "clean", "reset"],
        backends,
        ["list-models", "list-tests", "download"],
        *family_targets.values(),
        ["test-all"],
    ]
    phony_lines = " \\\n\t\t".join(" ".join(g) for g in groups if g)

    lines: list[str] = []
    lines.append("")
    lines.append(f"PY := {py_var}")
    lines.append("")
    lines.append(f".PHONY: {phony_lines}")
    lines.append("")
    lines.append("help:")
    lines.append('\t@echo "Available targets (frontend for $(PY)):"')
    lines.append('\t@echo ""')
    lines.append('\t@echo "  Setup:"')
    lines.append('\t@echo "    sync         - uv sync dependencies"')
    lines.append('\t@echo "    info         - show cyllama backend info"')
    lines.append('\t@echo "    clean        - remove .venv"')
    lines.append('\t@echo "    reset        - clean + sync"')
    for b in backends:
        dist = BACKENDS[b]
        lines.append(f'\t@echo "    {b:<12} - install {dist}"')
    lines.append('\t@echo ""')
    lines.append('\t@echo "  Models:"')
    lines.append('\t@echo "    list-models  - list known models and whether they are on disk"')
    lines.append('\t@echo "    download     - download all known models (use $(PY) download <key> for one)"')

    for fam, mapping in TEST_FAMILIES.items():
        title = FAMILY_TITLES.get(fam, fam)
        lines.append('\t@echo ""')
        lines.append(f'\t@echo "  {title} tests (backend auto-detected):"')
        for n in sorted(mapping):
            doc = (mapping[n].__doc__ or "").strip().rstrip(".")
            label = f"test-{fam}-{n}"
            lines.append(f'\t@echo "    {label:<{width}}- {doc}"')
        label = f"test-{fam}-all"
        lines.append(f'\t@echo "    {label:<{width}}- run all {fam} tests"')

    lines.append('\t@echo ""')
    lines.append('\t@echo "    list-tests   - list available smoke tests"')
    lines.append('\t@echo "    test-all     - run every test in every family"')

    def rule(target: str, args: str) -> None:
        lines.append("")
        lines.append(f"{target}:")
        lines.append(f"\t@$(PY) {args}")

    rule("sync", "sync")
    rule("info", "info")
    rule("clean", "clean")
    rule("reset", "reset")
    for b in backends:
        rule(b, f"install {b}")
    rule("list-models", "list-models")
    rule("list-tests", "list-tests")
    rule("download", "download all")
    for target in test_targets():
        if target != "test-all":
            rule(target, target)
    rule("test-all", "test-all")
    lines.append("")
    return "\n".join(lines)


def cmd_gen_makefile(args: argparse.Namespace) -> int:
    content = _render_makefile()
    if args.output:
        Path(args.output).write_text(content)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(content)
    return 0


def cmd_list_tests(_args: argparse.Namespace) -> int:
    width = max(len(t) for t in test_targets())
    for target, (kind, n) in test_targets().items():
        if kind == "all":
            doc = "every test in every family"
        elif n == "all":
            doc = f"all {kind} tests"
        else:
            doc = (TEST_FAMILIES[kind][n].__doc__ or "").strip()
        print(f"{target:<{width}}  {doc}")
    return 0


def test_targets() -> dict[str, tuple[str, str]]:
    """Map each ``test-*`` target name to the (family, case) it runs.

    One token per test -- ``test-all``, ``test-gen-all``, ``test-sd-3`` -- so
    the CLI and the generated Makefile name the same things.
    """
    targets: dict[str, tuple[str, str]] = {"test-all": ("all", "all")}
    for fam, mapping in TEST_FAMILIES.items():
        for n in sorted(mapping):
            targets[f"test-{fam}-{n}"] = (fam, n)
        targets[f"test-{fam}-all"] = (fam, "all")
    return targets


def _collect_runs(kind: str, n: str) -> list[tuple[str, str]]:
    """Expand ('all'|<family>, 'all'|'1'|...) into concrete (kind, n) pairs."""
    kinds = list(TEST_FAMILIES) if kind == "all" else [kind]
    runs: list[tuple[str, str]] = []
    for k in kinds:
        mapping = TEST_FAMILIES[k]
        if n == "all":
            runs.extend((k, nk) for nk in sorted(mapping))
        elif n in mapping:
            runs.append((k, n))
        elif kind != "all":
            # An explicit `test embed 3` is a mistake worth reporting; the same
            # number under `test all 3` just means "the families that have a 3".
            print(
                f"error: no test '{n}' in family '{k}' (have: {', '.join(sorted(mapping))})",
                file=sys.stderr,
            )
            sys.exit(2)
    if not runs:
        print(f"error: no tests matched kind={kind} n={n}", file=sys.stderr)
        sys.exit(2)
    return runs


def _use_color(no_color: bool) -> bool:
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def preflight(backend: str) -> str | None:
    """Import cyllama once up front; return an error message, or None if fine.

    Every test shells out through `uv run`, which re-syncs the project first.
    Against an installed wheel that is a no-op. In an editable checkout it
    rebuilds the extension -- but only when the *sources* changed, never
    because the environment did, so an extension previously built for another
    backend is reused as-is. Linked against this backend's static libs it then
    fails to import, and without this check that arrives once per test as an
    `undefined symbol` traceback with no hint of the cause.
    """
    proc = subprocess.run(
        [*python_cmd(), "-c", "import cyllama"],
        cwd=ROOT,
        env={**os.environ, **env_for(backend)},
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return None
    detail = (proc.stderr or proc.stdout).strip().splitlines()
    tail = detail[-1] if detail else f"exit code {proc.returncode}"
    hint = ""
    if VENV is not None:
        hint = (
            f"\n  Environment under test: {venv_python(VENV)}"
            f"\n  Install a wheel into it:  --venv {VENV} --wheel <path-to-wheel>"
            f"\n  ...or from PyPI:          --venv {VENV} --pypi --backend {backend}"
        )
    elif "undefined symbol" in tail:
        env_key = next(iter(BACKEND_ENV_DEFAULTS.get(backend, {})), None)
        if env_key:
            hint = (
                f"\n  The installed cyllama was not built for '{backend}'. In an editable"
                f"\n  checkout, rebuild it:  {env_key}=1 uv pip install -e ."
            )
    return f"cannot import cyllama: {tail}{hint}"


def maybe_install(args: argparse.Namespace) -> int:
    """Install before testing when --wheel/--pypi was given; no-op otherwise."""
    spec = resolve_install_spec(args)
    extra = getattr(args, "extra", None)
    if spec is None:
        if VENV is not None:
            ensure_venv(VENV)  # so a bare --venv still produces a usable env
        if extra:
            return pip_install([], extra=extra)
        return 0
    return pip_install(
        spec,
        upgrade=getattr(args, "upgrade", False),
        reinstall=getattr(args, "reinstall", False),
        extra=getattr(args, "extra", None),
    )


def cmd_test(args: argparse.Namespace) -> int:
    # --dry-run promises to touch nothing, so it precedes the install step.
    if args.dry_run:
        backend = getattr(args, "backend", None) or detect_backend() or "?"
        for k, n in _collect_runs(args.kind, args.n):
            print(f"would run: {k} {n} (backend={backend})")
        return 0

    rc = maybe_install(args)
    if rc != 0:
        return rc
    backend = require_backend(getattr(args, "backend", None))
    runs = _collect_runs(args.kind, args.n)

    problem = preflight(backend)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 1

    color = _use_color(args.no_color)
    green = "\033[32m" if color else ""
    red = "\033[31m" if color else ""
    reset = "\033[0m" if color else ""

    results: list[tuple[str, str, int, float]] = []
    for k, n in runs:
        mapping = TEST_FAMILIES[k]
        print(f"\n=== {k} test {n} (backend={backend}) ===")
        started = time.monotonic()
        try:
            rc = mapping[n](backend, args.timeout)
        except ModelSourceUnavailable as e:
            print(f"skip: {e}", file=sys.stderr)
            rc = 2
        results.append((k, n, rc, time.monotonic() - started))
        if rc != 0 and args.fail_fast:
            break

    # Summary
    print("\n=== summary ===")
    worst = 0
    for k, n, rc, secs in results:
        status = f"{green}PASS{reset}" if rc == 0 else f"{red}FAIL (rc={rc}){reset}"
        print(f"  {k} {n}: {status}  ({secs:.1f}s)")
        worst = max(worst, rc)
    passed = sum(1 for r in results if r[2] == 0)
    total = sum(r[3] for r in results)
    print(f"{passed}/{len(results)} passed in {total:.1f}s")
    return worst


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------


def _common_parser() -> argparse.ArgumentParser:
    """Options accepted both before and after the subcommand."""
    c = argparse.ArgumentParser(add_help=False)
    c.add_argument(
        "--venv",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="virtualenv to test against; created if missing. Every subprocess "
        "runs this interpreter directly instead of `uv run`, so the installed "
        "wheel is what gets tested even from inside the source checkout.",
    )
    c.add_argument(
        "--models-dir",
        "--models_dir",
        metavar="PATH",
        dest="models_dir",
        default=argparse.SUPPRESS,
        help=f"directory holding the GGUF/safetensors models (default: {MODELS_DIR})",
    )
    shorthand = c.add_mutually_exclusive_group()
    for backend in BACKENDS:
        shorthand.add_argument(
            f"--{backend}",
            dest="backend_shorthand",
            action="store_const",
            const=backend,
            default=argparse.SUPPRESS,
            help=f"shorthand for --venv {DEFAULT_VENV_PREFIX}{backend} --backend {backend}",
        )
    c.add_argument(
        "--backend",
        choices=list(BACKENDS),
        default=argparse.SUPPRESS,
        help="backend under test; auto-detected from the installed distribution",
    )
    c.add_argument(
        "--python",
        metavar="VERSION",
        default=argparse.SUPPRESS,
        help="interpreter for a venv created by --venv (e.g. 3.12); passed to `uv venv --python`",
    )
    c.add_argument(
        "--data-dir",
        "--data_dir",
        metavar="PATH",
        dest="data_dir",
        default=argparse.SUPPRESS,
        help=f"directory holding corpus1.txt / jfk.wav (default: {DATA_DIR})",
    )
    c.add_argument(
        "--wheel",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="local wheel file to install into --venv before running",
    )
    c.add_argument(
        "--pypi",
        nargs="?",
        const=True,
        default=argparse.SUPPRESS,
        metavar="SPEC",
        help="install from PyPI instead of a local wheel. Bare --pypi installs "
        "the distribution for --backend; or name one, e.g. --pypi cyllama-cuda12==0.4.3. "
        "Because the spec is optional, a bare --pypi must not directly precede the "
        "subcommand -- put it last, or give it a spec.",
    )
    c.add_argument(
        "--with",
        dest="extra",
        action="append",
        metavar="PKG",
        default=argparse.SUPPRESS,
        help="extra package to install alongside the wheel (repeatable), "
        "e.g. --with numpy when diagnosing a wheel that predates a fix.",
    )
    c.add_argument(
        "--upgrade",
        action="store_true",
        default=argparse.SUPPRESS,
        help="pass --upgrade to uv pip install",
    )
    c.add_argument(
        "--reinstall",
        action="store_true",
        default=argparse.SUPPRESS,
        help="pass --reinstall to uv pip install",
    )
    return c


def build_parser() -> argparse.ArgumentParser:
    common = _common_parser()
    p = argparse.ArgumentParser(
        description="cyllama wheel tester",
        parents=[common],
        epilog=(
            "example: run_wheel_test.py --venv .venv-vulkan "
            "--wheel dist/cyllama_vulkan-0.4.3-cp312-abi3-win_amd64.whl "
            "--models-dir models test all all"
        ),
    )
    _sub = p.add_subparsers(dest="cmd", required=True)

    class sub:  # noqa: N801 - thin shim so add_parser always inherits `common`
        @staticmethod
        def add_parser(name, parents=(), **kw):
            return _sub.add_parser(name, parents=[common, *parents], **kw)

    sub.add_parser("info", help="show python/backend/models info").set_defaults(func=cmd_info)
    sub.add_parser("sync", help="uv sync project dependencies").set_defaults(func=cmd_sync)
    sub.add_parser("clean", help="remove the .venv directory").set_defaults(func=cmd_clean)
    sub.add_parser("reset", help="clean + sync").set_defaults(func=cmd_reset)

    inst = sub.add_parser("install", help="install a cyllama backend (PyPI, or --wheel)")
    inst.add_argument("backend", nargs="?", choices=list(BACKENDS), default=argparse.SUPPRESS)
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

    gm = sub.add_parser("gen-makefile", help="generate the Makefile from this script's registries")
    gm.add_argument("-o", "--output", help="write to file instead of stdout (e.g. -o Makefile)")
    gm.set_defaults(func=cmd_gen_makefile)

    # One subcommand per test target. `test-sd-3` rather than `test sd 3`, so a
    # target is a single token and matches the Makefile rule of the same name.
    test_opts = argparse.ArgumentParser(add_help=False)
    test_opts.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-test timeout in seconds (default: no timeout)",
    )
    test_opts.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop at the first failing test instead of running the full matrix",
    )
    test_opts.add_argument(
        "--dry-run",
        action="store_true",
        help="print the test matrix without downloading or invoking anything",
    )
    test_opts.add_argument(
        "--no-color",
        action="store_true",
        help="disable colored PASS/FAIL output in the summary",
    )

    for target, (kind, n) in test_targets().items():
        if kind == "all":
            blurb = "run every test in every family"
        elif n == "all":
            blurb = f"run all {kind} tests"
        else:
            blurb = (TEST_FAMILIES[kind][n].__doc__ or "").strip().rstrip(".")
        t = sub.add_parser(target, parents=[test_opts], help=blurb)
        t.set_defaults(func=cmd_test, kind=kind, n=n)

    return p


def _apply_cli_overrides(args: argparse.Namespace) -> None:
    global VENV, MODELS_DIR, VENV_PYTHON, DATA_DIR
    if getattr(args, "venv", None):
        VENV = Path(args.venv).expanduser().resolve()
    elif getattr(args, "backend_shorthand", None):
        # --cuda etc. only fills in what was not given explicitly, so
        # `--cuda --venv /tmp/x` still targets /tmp/x.
        VENV = (ROOT / f"{DEFAULT_VENV_PREFIX}{args.backend_shorthand}").resolve()
    if getattr(args, "backend_shorthand", None) and not getattr(args, "backend", None):
        args.backend = args.backend_shorthand
    if getattr(args, "python", None):
        VENV_PYTHON = args.python
    if getattr(args, "models_dir", None):
        MODELS_DIR = Path(args.models_dir).expanduser().resolve()
    if getattr(args, "data_dir", None):
        DATA_DIR = Path(args.data_dir).expanduser().resolve()


def main() -> None:
    _apply_env_overrides()
    args = build_parser().parse_args()
    _apply_cli_overrides(args)
    rc = args.func(args)
    sys.exit(int(rc or 0))


if __name__ == "__main__":
    main()
