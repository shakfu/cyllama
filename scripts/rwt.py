#!/usr/bin/env python3
"""Self-contained smoke-test runner for built cyllama wheels.

``--venv`` names the environment under test and every subprocess runs that
interpreter directly. Without it the script falls back to ``uv run``, which
re-syncs whichever project owns the cwd -- so from the cyllama checkout it
would build the extension from source and test that, never the wheel.

``--cuda`` (and ``--cpu`` / ``--metal`` / ``--vulkan`` / ``--rocm`` /
``--sycl``) names the backend and points ``--venv`` at ``.venv-<backend>``; an
explicit ``--venv`` wins. Without one the backend is detected from what the
venv has installed. ``--metal`` and ``--cpu`` install the same ``cyllama``
distribution -- CI builds it with Metal on macOS and without it elsewhere --
so a bare ``cyllama`` in a venv is reported as ``metal`` on macOS.

``install`` is the only subcommand that writes to the venv: ``--wheel`` says
what to put there -- a local wheel or a requirement for the index, told apart
by shape -- and creating the venv is part of it. Every test target expects an
environment that already has cyllama in it.

``test`` takes one target -- ``test-all``, ``test-gen-all``, ``test-sd-3`` --
named identically to the generated Makefile rules; ``list tests`` prints them.

The script is organised as a handful of collaborating objects rather than
module state: :class:`Paths` resolves the directory layout, :class:`Env` owns
the environment under test (venv, backend, subprocesses), :class:`ModelRegistry`
knows where models and data assets come from, :class:`TestSuite` holds the test
cases, and :class:`Cli` wires them to argparse.

Examples:
    # create .venv-cuda and install the latest cyllama-cuda12 from the index;
    # the backend names the distribution, so --wheel is not needed here
    python rwt.py install --cuda
    python rwt.py install --metal          # macOS: the plain `cyllama` wheel

    # --wheel is only for pinning a version or naming a local artifact
    python rwt.py install --cuda --wheel cyllama-cuda12==0.4.2
    python rwt.py install --vulkan --wheel dist/cyllama_vulkan-0.4.3-cp312-abi3-win_amd64.whl

    # run everything, one family, or one case
    python rwt.py test --cuda test-all
    python rwt.py test --cuda test-rag-all
    python rwt.py test --cuda test-sd-3 --timeout 600

    # against a venv somewhere else; the backend is detected from what is
    # installed, so no --cuda/--vulkan/... is needed
    python rwt.py test --venv /tmp/wheel-check test-all

    # show the matrix without downloading or running anything
    python rwt.py test --cuda test-all --dry-run

    # environment, registry and target listings
    python rwt.py info --cuda
    python rwt.py list
    python rwt.py download all --models-dir models
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_NAME = Path(__file__).name


# ---------------------------------------------------------------------------
# exceptions
# ---------------------------------------------------------------------------


class ModelSourceUnavailable(RuntimeError):
    """Raised when a model has no configured source and isn't on disk."""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------


@dataclass
class Paths:
    """The directory layout every other object resolves against."""

    root: Path
    models_dir: Path
    data_dir: Path

    # The checkout keeps text under tests/media but audio under tests/samples,
    # so look in both rather than making the caller pick one with --data-dir.
    data_fallback_names: tuple[str, ...] = ("tests/media", "tests/samples")

    @staticmethod
    def find_root() -> Path:
        """Locate the project root: the cwd for subprocesses and the parent of
        ``models/`` and ``.venv/``.

        This file is checked in as ``<repo>/scripts/rwt.py`` but is also
        meant to be copied out standalone (as ``./rwt.py``) into a bare
        uv-managed wheel-test directory. Walking up to the nearest project
        marker handles both layouts; using ``__file__``'s own directory would
        resolve to ``<repo>/scripts`` in-repo and download models to
        ``scripts/models``.
        """
        here = Path(__file__).resolve().parent
        for candidate in (here, *here.parents):
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                return candidate
        return here

    @classmethod
    def from_environ(cls) -> Paths:
        root = cls.find_root()
        return cls(
            root=root,
            models_dir=Path(os.environ.get("CYLLAMA_MODELS_DIR", root / "models")),
            data_dir=Path(os.environ.get("CYLLAMA_DATA_DIR", root / "tests" / "media")),
        )

    @property
    def data_dirs(self) -> list[Path]:
        return [self.data_dir, *(self.root / name for name in self.data_fallback_names)]

    def find_data_asset(self, name: str) -> Path | None:
        """First existing copy of `name` in --data-dir or the checkout's data dirs."""
        for d in self.data_dirs:
            candidate = d / name
            if candidate.exists():
                return candidate
        return None


# ---------------------------------------------------------------------------
# the environment under test
# ---------------------------------------------------------------------------


class Env:
    """The environment cyllama is tested in: its venv, backend and subprocesses.

    When ``venv`` is set, every subprocess runs that interpreter *directly*
    rather than through ``uv run``. This matters: ``uv run`` re-syncs whichever
    project owns the cwd, so run from this checkout it would build cyllama from
    source and test that instead of the installed wheel. ``venv=None`` restores
    the legacy ``uv run`` behaviour.
    """

    # Backend -> distribution on PyPI. Only the GPU backends get a renamed
    # distribution; `cpu` and `metal` are both the plain `cyllama` wheel, which
    # CI builds with GGML_METAL=1 on macOS and GGML_METAL=0 everywhere else.
    BACKENDS: dict[str, str] = {
        "cpu": "cyllama",
        "metal": "cyllama",
        "cuda": "cyllama-cuda12",
        "vulkan": "cyllama-vulkan",
        "rocm": "cyllama-rocm",
        "sycl": "cyllama-sycl",
    }

    # Distribution -> backend, for detection. Inverting BACKENDS would be
    # ambiguous for `cyllama`, so resolve that one by platform: the macOS wheel
    # is the Metal wheel, and there is no CPU-only macOS wheel to confuse it with.
    DISTRIBUTIONS: dict[str, str] = {
        **{dist: b for b, dist in BACKENDS.items() if dist != "cyllama"},
        "cyllama": "metal" if sys.platform == "darwin" else "cpu",
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

    _DETECT_SRC = """
import importlib.metadata as md
for dist, backend in {distributions!r}.items():
    try:
        md.distribution(dist)
        print(backend)
        break
    except md.PackageNotFoundError:
        pass
"""

    def __init__(
        self,
        paths: Paths,
        venv: Path | None = None,
        venv_python_version: str | None = None,
        uv: str | None = None,
    ) -> None:
        self.paths = paths
        # The venv under test; None means fall back to `uv run`.
        self.venv = venv
        # Interpreter `uv venv` should build the target env from (--python).
        # Left unset, uv picks its own default, which is not necessarily the
        # version a given wheel was built for.
        self.venv_python_version = venv_python_version
        # Resolve `uv` once. Everything this script shells out to Python for is
        # routed through `uv run` so it executes inside the project's uv venv
        # regardless of how the script itself was launched.
        self.uv = uv or shutil.which("uv") or "uv"

    # -- venv plumbing ------------------------------------------------------

    @staticmethod
    def venv_python(venv: Path) -> Path:
        """Interpreter path inside `venv`, on either the Windows or POSIX layout."""
        win = venv / "Scripts" / "python.exe"
        if win.exists():
            return win
        posix = venv / "bin" / "python"
        if posix.exists():
            return posix
        return win if os.name == "nt" else posix

    def ensure_venv(self, venv: Path) -> Path:
        """Create `venv` if it does not exist yet; return its interpreter."""
        py = self.venv_python(venv)
        if not py.exists():
            print(f"creating venv at {venv}")
            cmd = [self.uv, "venv", str(venv)]
            if self.venv_python_version:
                cmd += ["--python", self.venv_python_version]
            subprocess.run(cmd, check=True)
            py = self.venv_python(venv)
        return py

    def python_cmd(self) -> list[str]:
        """argv prefix that runs Python in the environment under test."""
        if self.venv is not None:
            return [str(self.venv_python(self.venv))]
        return [self.uv, "run", "python"]

    def pip_install(
        self,
        spec: list[str],
        upgrade: bool = False,
        reinstall: bool = False,
        extra: list[str] | None = None,
    ) -> int:
        """Install `spec` (plus any --with packages) into the environment under test."""
        cmd = [self.uv, "pip", "install"]
        if self.venv is not None:
            cmd += ["--python", str(self.ensure_venv(self.venv))]
        if upgrade:
            cmd.append("--upgrade")
        if reinstall:
            cmd.append("--reinstall")
        return self.run(cmd + spec + list(extra or []))

    # -- subprocesses -------------------------------------------------------

    @staticmethod
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
        self,
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
        proc = subprocess.Popen(cmd, cwd=self.paths.root, env=full_env, start_new_session=os.name != "nt")
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"error: command timed out after {timeout}s", file=sys.stderr)
            self._kill_tree(proc)
            rc = 124  # conventional timeout exit code
        if check and rc != 0:
            sys.exit(rc)
        return rc

    def cyllama(self, argv: list[str], env: dict[str, str] | None = None, timeout: float | None = None) -> int:
        return self.run([*self.python_cmd(), "-m", "cyllama", *argv], env=env, timeout=timeout)

    def cyllama_module(
        self,
        module: str,
        argv: list[str],
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> int:
        return self.run([*self.python_cmd(), "-m", module, *argv], env=env, timeout=timeout)

    def has_module(self, name: str) -> bool:
        """Whether `name` is importable in the environment under test."""
        proc = subprocess.run(
            [*self.python_cmd(), "-c", f"import {name}"],
            cwd=self.paths.root,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0

    # -- backend detection --------------------------------------------------

    def _detect_backend_in_venv(self, venv: Path) -> str | None:
        py = self.venv_python(venv)
        if not py.exists():
            return None
        proc = subprocess.run(
            [str(py), "-c", self._DETECT_SRC.format(distributions=self.DISTRIBUTIONS)],
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or None

    def detect_backend(self) -> str | None:
        # With an explicit target venv, ask *it* what is installed. importlib.metadata
        # here would describe the interpreter running this script, which under
        # `uv run` from the checkout is the project env, not the wheel under test.
        if self.venv is not None:
            return self._detect_backend_in_venv(self.venv)
        for dist, backend in self.DISTRIBUTIONS.items():
            try:
                md.distribution(dist)
                return backend
            except md.PackageNotFoundError:
                continue
        return None

    def env_for(self, backend: str) -> dict[str, str]:
        """Return default env overrides for a backend, skipping keys the
        caller has already set in the surrounding environment."""
        defaults = self.BACKEND_ENV_DEFAULTS.get(backend, {})
        return {k: v for k, v in defaults.items() if k not in os.environ}

    def require_backend(self, requested: str | None) -> str:
        detected = self.detect_backend()
        if requested and detected and requested != detected:
            print(
                f"warning: requested backend '{requested}' but '{detected}' is installed",
                file=sys.stderr,
            )
        backend = requested or detected
        if not backend:
            flags = ",".join("--" + b for b in self.BACKENDS)
            if self.venv is not None:
                print(
                    f"error: no cyllama backend installed in {self.venv}."
                    f"\n  Install from the index: {SCRIPT_NAME} install --venv {self.venv} {{{flags}}}"
                    f"\n  ...or a local wheel:    {SCRIPT_NAME} install --venv {self.venv} --wheel <path>",
                    file=sys.stderr,
                )
            else:
                print(
                    f"error: no cyllama backend installed. Run: {SCRIPT_NAME} install {{{flags}}}",
                    file=sys.stderr,
                )
            sys.exit(2)
        return backend

    def preflight(self, backend: str) -> str | None:
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
            [*self.python_cmd(), "-c", "import cyllama"],
            cwd=self.paths.root,
            env={**os.environ, **self.env_for(backend)},
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return None
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        tail = detail[-1] if detail else f"exit code {proc.returncode}"
        hint = ""
        if self.venv is not None:
            hint = (
                f"\n  Environment under test: {self.venv_python(self.venv)}"
                f"\n  Install from the index:   {SCRIPT_NAME} install --venv {self.venv} --{backend}"
                f"\n  ...or a local wheel:      {SCRIPT_NAME} install --venv {self.venv} --wheel <path-to-wheel>"
            )
        elif "undefined symbol" in tail:
            env_key = next(iter(self.BACKEND_ENV_DEFAULTS.get(backend, {})), None)
            if env_key:
                hint = (
                    f"\n  The installed cyllama was not built for '{backend}'. In an editable"
                    f"\n  checkout, rebuild it:  {env_key}=1 uv pip install -e ."
                )
        return f"cannot import cyllama: {tail}{hint}"


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


class ModelRegistry:
    """Known models and data assets, and how to get them onto disk."""

    JFK_WAV_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"

    # Which tests need which models.
    SD_REQUIREMENTS: list[str] = ["z-image-turbo", "ae", "qwen3-4b"]
    RAG_REQUIREMENTS: list[str] = ["qwen3-4b", "bge-small-en"]

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

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.sources = self.default_sources()
        self.apply_env_overrides()

    @staticmethod
    def default_sources() -> dict[str, ModelSource]:
        """Best-effort defaults -- overridable via CYLLAMA_MODEL_<KEY>=repo_id:file
        or by placing files in the models dir yourself. Use `list-models` to inspect.
        """
        return {
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

    def apply_env_overrides(self) -> None:
        """Allow overriding repo ids via env vars (CYLLAMA_MODEL_<KEY>=repo:file)."""
        for key, src in self.sources.items():
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

    # -- downloads ----------------------------------------------------------

    @staticmethod
    def download_urllib(url: str, dest: Path) -> None:
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

    @staticmethod
    def download_hf(repo_id: str, filename: str, dest: Path) -> None:
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
        # Land the file directly in the models dir rather than copying from the
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

    # -- lookups ------------------------------------------------------------

    def path_for(self, key: str) -> Path:
        return self.paths.models_dir / self.sources[key].filename

    def ensure_model(self, key: str) -> Path:
        src = self.sources[key]
        dest = self.path_for(key)
        if dest.exists():
            return dest
        if src.url:
            self.download_urllib(src.url, dest)
        elif src.repo_id:
            self.download_hf(src.repo_id, src.hub_filename(), dest)
        else:
            raise ModelSourceUnavailable(f"no source configured for model '{key}' ({src.filename}). {src.notes}")
        return dest

    def ensure_models(self, keys: list[str]) -> dict[str, Path]:
        return {k: self.ensure_model(k) for k in keys}

    # -- data assets --------------------------------------------------------
    #
    # These are inputs rather than models. In the checkout they already exist
    # under tests/media; standalone they do not, so each has a fallback --
    # jfk.wav is fetched from whisper.cpp, and the corpus is synthesised rather
    # than downloaded, since the one in the repo is a copyrighted short story.

    def ensure_corpus(self) -> Path:
        """Path to a line-per-text corpus, preferring the checkout's own."""
        repo_copy = self.paths.find_data_asset("corpus1.txt")
        if repo_copy is not None:
            return repo_copy
        generated = self.paths.models_dir / "corpus_generated.txt"
        if not generated.exists():
            print(f"writing generated corpus -> {generated}")
            generated.parent.mkdir(parents=True, exist_ok=True)
            generated.write_text("\n".join(self.GENERATED_CORPUS) + "\n", encoding="utf-8")
        return generated

    def ensure_audio(self) -> Path:
        """Path to the jfk.wav sample, downloading it if the checkout lacks one."""
        repo_copy = self.paths.find_data_asset("jfk.wav")
        if repo_copy is not None:
            return repo_copy
        dest = self.paths.models_dir / "jfk.wav"
        if not dest.exists():
            self.download_urllib(self.JFK_WAV_URL, dest)
        return dest


# ---------------------------------------------------------------------------
# tests (inlined from the shell scripts in ~/projects/demo/scripts)
# ---------------------------------------------------------------------------

TestFn = Callable[[str, "float | None"], int]


class TestSuite:
    """The smoke-test cases, grouped into families.

    Every case has the same signature -- ``(backend, timeout) -> exit code`` --
    and its docstring is the one-line description `list tests` and the generated
    Makefile print, so keep them short.
    """

    # Every test family, in the order `test-all` runs them: cheap and
    # fast-failing first, the multi-minute image cases last.
    FAMILY_ORDER: tuple[str, ...] = ("embed", "transcribe", "gen", "rag", "sd")

    # Human-readable section headings for the generated Makefile's help text.
    FAMILY_TITLES: dict[str, str] = {
        "embed": "Embedding",
        "transcribe": "Transcription",
        "gen": "Generation",
        "rag": "RAG",
        "sd": "Stable Diffusion",
    }

    def __init__(self, env: Env, models: ModelRegistry) -> None:
        self.env = env
        self.models = models
        self.families: dict[str, dict[str, TestFn]] = {
            "embed": {"1": self.embed_1},
            "transcribe": {"1": self.transcribe_1},
            "gen": {"1": self.gen_1, "2": self.gen_2, "3": self.gen_3},
            "rag": {"1": self.rag_1, "2": self.rag_2},
            "sd": {"1": self.sd_1, "2": self.sd_2, "3": self.sd_3},
        }
        # Declared separately from FAMILY_ORDER so a family added to one and not
        # the other is caught here rather than silently skipped by `test-all`.
        assert tuple(self.families) == self.FAMILY_ORDER, "families must match FAMILY_ORDER"

    # -- stable diffusion ---------------------------------------------------

    def sd_1(self, backend: str, timeout: float | None) -> int:
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
        paths = self.models.ensure_models(ModelRegistry.SD_REQUIREMENTS)
        return self.env.cyllama_module(
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
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def sd_2(self, backend: str, timeout: float | None) -> int:
        """z_turbo cpu-offload."""
        paths = self.models.ensure_models(ModelRegistry.SD_REQUIREMENTS)
        return self.env.cyllama_module(
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
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def sd_3(self, backend: str, timeout: float | None) -> int:
        """z_turbo cpu-offload + flash-attn."""
        paths = self.models.ensure_models(ModelRegistry.SD_REQUIREMENTS)
        return self.env.cyllama_module(
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
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    # -- generation ---------------------------------------------------------

    def gen_1(self, backend: str, timeout: float | None) -> int:
        """Llama-3.2-1B short prompt."""
        model = self.models.ensure_model("llama-3.2-1b")
        return self.env.cyllama(
            [
                "gen",
                "-m",
                str(model),
                "-p",
                "Explain quantum entanglement in one paragraph.",
                "-n",
                "256",
                "--stats",
            ],
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def gen_2(self, backend: str, timeout: float | None) -> int:
        """Qwen3-4B streamed."""
        model = self.models.ensure_model("qwen3-4b")
        return self.env.cyllama(
            ["gen", "-m", str(model), "-p", "Write a haiku about GPUs.", "-n", "256", "--stream", "--stats"],
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def gen_3(self, backend: str, timeout: float | None) -> int:
        """Gemma-4-E4B streamed."""
        model = self.models.ensure_model("gemma-e4b")
        return self.env.cyllama(
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
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    # -- embedding ----------------------------------------------------------

    def embed_1(self, backend: str, timeout: float | None) -> int:
        """corpus similarity ranking."""
        model = self.models.ensure_model("bge-small-en")
        corpus = self.models.ensure_corpus()
        return self.env.cyllama(
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
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    # -- transcription ------------------------------------------------------

    def transcribe_1(self, backend: str, timeout: float | None) -> int:
        """jfk.wav speech-to-text."""
        # The invariant is that transcription works on a bare wheel install: the
        # wheels declare no dependencies, so nothing on this path may import a
        # third-party package. Do not gate on numbers being present -- gating on
        # numpy would fail a *correctly* built wheel, which is the whole point.
        model = self.models.ensure_model("whisper-base-en")
        audio = self.models.ensure_audio()
        rc = self.env.cyllama(
            ["transcribe", "-f", str(audio), "-m", str(model)],
            env=self.env.env_for(backend),
            timeout=timeout,
        )
        if rc != 0 and not self.env.has_module("numpy"):
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

    # -- rag ----------------------------------------------------------------

    def rag_1(self, backend: str, timeout: float | None) -> int:
        """in-memory index + query."""
        paths = self.models.ensure_models(ModelRegistry.RAG_REQUIREMENTS)
        corpus = self.models.ensure_corpus()
        return self.env.cyllama(
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
            env=self.env.env_for(backend),
            timeout=timeout,
        )

    def rag_2(self, backend: str, timeout: float | None) -> int:
        """persistent sqlite vector store (build + reopen)."""
        paths = self.models.ensure_models(ModelRegistry.RAG_REQUIREMENTS)
        corpus = self.models.ensure_corpus()
        db = self.env.paths.root / "vector.db"
        if db.exists():
            db.unlink()  # start from nothing so the create path is covered

        def query(prompt: str) -> int:
            return self.env.cyllama(
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
                env=self.env.env_for(backend),
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

    # -- target bookkeeping -------------------------------------------------

    def targets(self) -> dict[str, tuple[str, str]]:
        """Map each ``test-*`` target name to the (family, case) it runs.

        One token per test -- ``test-all``, ``test-gen-all``, ``test-sd-3`` -- so
        the CLI and the generated Makefile name the same things.
        """
        targets: dict[str, tuple[str, str]] = {"test-all": ("all", "all")}
        for fam, mapping in self.families.items():
            for n in sorted(mapping):
                targets[f"test-{fam}-{n}"] = (fam, n)
            targets[f"test-{fam}-all"] = (fam, "all")
        return targets

    def describe(self, kind: str, n: str) -> str:
        """One-line description of a target, from the case's docstring."""
        if kind == "all":
            return "every test in every family"
        if n == "all":
            return f"all {kind} tests"
        return (self.families[kind][n].__doc__ or "").strip()

    def collect_runs(self, kind: str, n: str) -> list[tuple[str, str]]:
        """Expand ('all'|<family>, 'all'|'1'|...) into concrete (kind, n) pairs."""
        kinds = list(self.families) if kind == "all" else [kind]
        runs: list[tuple[str, str]] = []
        for k in kinds:
            mapping = self.families[k]
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

    def run_case(self, kind: str, n: str, backend: str, timeout: float | None) -> int:
        return self.families[kind][n](backend, timeout)


# ---------------------------------------------------------------------------
# generated Makefile
# ---------------------------------------------------------------------------


class MakefileRenderer:
    """Renders the Makefile whose rules mirror this script's own targets."""

    PY_VAR = "uv run ./rwt.py"

    def __init__(self, env: Env, suite: TestSuite) -> None:
        self.env = env
        self.suite = suite
        self.lines: list[str] = []

    def render(self) -> str:
        self.lines = []
        backends = list(self.env.BACKENDS)

        family_targets: dict[str, list[str]] = {
            fam: [f"test-{fam}-{n}" for n in sorted(mapping)] + [f"test-{fam}-all"]
            for fam, mapping in self.suite.families.items()
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

        add = self.lines.append
        add("")
        add(f"PY := {self.PY_VAR}")
        add("")
        add(f".PHONY: {phony_lines}")
        add("")
        add("help:")
        add('\t@echo "Available targets (frontend for $(PY)):"')
        add('\t@echo ""')
        add('\t@echo "  Setup:"')
        add('\t@echo "    sync         - uv sync dependencies"')
        add('\t@echo "    info         - show cyllama backend info"')
        add('\t@echo "    clean        - remove .venv"')
        add('\t@echo "    reset        - clean + sync"')
        for b in backends:
            dist = self.env.BACKENDS[b]
            add(f'\t@echo "    {b:<12} - install {dist}"')
        add('\t@echo ""')
        add('\t@echo "  Models:"')
        add('\t@echo "    list-models  - list known models and whether they are on disk"')
        add('\t@echo "    download     - download all known models (use $(PY) download <key> for one)"')

        for fam, mapping in self.suite.families.items():
            title = self.suite.FAMILY_TITLES.get(fam, fam)
            add('\t@echo ""')
            add(f'\t@echo "  {title} tests (backend auto-detected):"')
            for n in sorted(mapping):
                doc = (mapping[n].__doc__ or "").strip().rstrip(".")
                label = f"test-{fam}-{n}"
                add(f'\t@echo "    {label:<{width}}- {doc}"')
            label = f"test-{fam}-all"
            add(f'\t@echo "    {label:<{width}}- run all {fam} tests"')

        add('\t@echo ""')
        add('\t@echo "    list         - list test targets and models"')
        add('\t@echo "    test-all     - run every test in every family"')

        self.rule("sync", "sync")
        self.rule("info", "info")
        self.rule("clean", "clean")
        self.rule("reset", "reset")
        for b in backends:
            self.rule(b, f"install --{b}")
        self.rule("list-models", "list models")
        self.rule("list-tests", "list tests")
        self.rule("download", "download all")
        for target in self.suite.targets():
            if target != "test-all":
                self.rule(target, f"test {target}")
        self.rule("test-all", "test test-all")
        add("")
        return "\n".join(self.lines)

    def rule(self, target: str, args: str) -> None:
        self.lines.append("")
        self.lines.append(f"{target}:")
        self.lines.append(f"\t@$(PY) {args}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class Cli:
    """Argparse wiring and the subcommand implementations.

    The parser is built against the *defaults* (so --help can quote them), then
    :meth:`configure` rebuilds the collaborators from what was actually parsed.
    """

    def __init__(self) -> None:
        self.paths = Paths.from_environ()
        # Where the --cpu/--cuda/--vulkan/... shorthands look for their venv.
        # Relative names resolve against root so the shorthand means the same
        # thing from any cwd.
        self.venv_prefix = os.environ.get("CYLLAMA_VENV_PREFIX", ".venv-")
        self.env = Env(self.paths)
        self.models = ModelRegistry(self.paths)
        self.suite = TestSuite(self.env, self.models)

    # -- configuration ------------------------------------------------------

    def configure(self, args: argparse.Namespace) -> None:
        """Apply parsed options; every collaborator is rebuilt from them."""
        if getattr(args, "models_dir", None):
            self.paths.models_dir = Path(args.models_dir).expanduser().resolve()
        if getattr(args, "data_dir", None):
            self.paths.data_dir = Path(args.data_dir).expanduser().resolve()

        venv: Path | None = None
        if getattr(args, "venv", None):
            venv = Path(args.venv).expanduser().resolve()
        elif getattr(args, "backend", None):
            # --cuda etc. only fills in what was not given explicitly, so
            # `--cuda --venv /tmp/x` still targets /tmp/x.
            venv = (self.paths.root / f"{self.venv_prefix}{args.backend}").resolve()
        self.env.venv = venv
        self.env.venv_python_version = getattr(args, "python", None) or None

    # -- simple commands ----------------------------------------------------

    def cmd_info(self, _args: argparse.Namespace) -> int:
        backend = self.env.detect_backend()
        venv = self.env.venv
        target = str(self.env.venv_python(venv)) if venv is not None else sys.executable
        print(f"{'python:':<9}{target}")
        print(f"{'backend:':<9}{backend or '(none)'}")
        print(f"{'models:':<9}{self.paths.models_dir}")
        if backend:
            self.env.cyllama(["info"])
        return 0

    def cmd_sync(self, _args: argparse.Namespace) -> int:
        return self.env.run([self.env.uv, "sync"])

    def cmd_clean(self, _args: argparse.Namespace) -> int:
        venv = self.env.venv if self.env.venv is not None else self.paths.root / ".venv"
        if venv.exists():
            print(f"removing {venv}")
            shutil.rmtree(venv)
        return 0

    def cmd_reset(self, args: argparse.Namespace) -> int:
        rc = self.cmd_clean(args)
        if rc != 0:
            return rc
        return self.cmd_sync(args)

    # -- install ------------------------------------------------------------

    @staticmethod
    def resolve_install_spec(args: argparse.Namespace) -> list[str] | None:
        """What ``--wheel`` asks to install, or None if it was not given.

        The value is either a local artifact or a requirement for the index, told
        apart by shape rather than by a second flag: a URL or anything carrying a
        path separator or a ``.whl`` suffix is a file, everything else is a spec
        handed to ``uv pip install`` as written (``cyllama-cuda12``,
        ``cyllama-vulkan==0.4.3``, ``cyllama-cuda12[extra]``).
        """
        value = args.wheel
        if not value:
            return None

        if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value):
            return [value]  # a URL; uv resolves it itself

        path = Path(value).expanduser()
        looks_local = path.suffix == ".whl" or path.exists() or "/" in value or "\\" in value
        if not looks_local:
            return [value]

        resolved = path.resolve()
        if not resolved.exists():
            print(f"error: wheel not found: {resolved}", file=sys.stderr)
            sys.exit(2)
        return [str(resolved)]

    def cmd_install(self, args: argparse.Namespace) -> int:
        spec = self.resolve_install_spec(args)
        if spec is None:
            # No --wheel: the backend names the distribution to fetch from the index.
            backend = getattr(args, "backend", None)
            if not backend:
                flags = "/".join("--" + b for b in self.env.BACKENDS)
                print(
                    f"error: give a backend ({flags}), or --wheel <path-or-spec>",
                    file=sys.stderr,
                )
                return 2
            spec = [self.env.BACKENDS[backend]]
        return self.env.pip_install(spec, upgrade=args.upgrade, reinstall=args.reinstall, extra=args.extra)

    # -- registries ---------------------------------------------------------

    def cmd_download(self, args: argparse.Namespace) -> int:
        keys = list(self.models.sources) if args.key == "all" else [args.key]
        failures = 0
        for k in keys:
            try:
                path = self.models.ensure_model(k)
                print(f"ok: {k} -> {path}")
            except ModelSourceUnavailable as e:
                print(f"skip: {k}: {e}", file=sys.stderr)
                failures += 1
        return 1 if failures else 0

    def cmd_list_models(self, _args: argparse.Namespace) -> int:
        for key, src in self.models.sources.items():
            source = f"hf:{src.repo_id}:{src.hub_filename()}" if src.repo_id else (src.url or "(no source configured)")
            on_disk = "YES" if (self.paths.models_dir / src.filename).exists() else "no"
            print(f"{key:<16} file={src.filename:<40} on_disk={on_disk:<3} source={source}")
            if src.notes and not src.repo_id and not src.url:
                print(f"{'':<16} note: {src.notes}")
        return 0

    def cmd_list_tests(self, _args: argparse.Namespace) -> int:
        targets = self.suite.targets()
        width = max(len(t) for t in targets)
        for target, (kind, n) in targets.items():
            print(f"{target:<{width}}  {self.suite.describe(kind, n)}")
        return 0

    def cmd_list(self, args: argparse.Namespace) -> int:
        """`list` with no argument shows both registries; `list tests|models` narrows."""
        what = getattr(args, "what", "all")
        rc = 0
        if what in ("tests", "all"):
            if what == "all":
                print("tests:")
            rc |= self.cmd_list_tests(args)
        if what in ("models", "all"):
            if what == "all":
                print("\nmodels:")
            rc |= self.cmd_list_models(args)
        return rc

    def cmd_gen_makefile(self, args: argparse.Namespace) -> int:
        content = MakefileRenderer(self.env, self.suite).render()
        if args.output:
            Path(args.output).write_text(content)
            print(f"wrote {args.output}")
        else:
            sys.stdout.write(content)
        return 0

    # -- test ---------------------------------------------------------------

    @staticmethod
    def _use_color(no_color: bool) -> bool:
        if no_color or os.environ.get("NO_COLOR"):
            return False
        return sys.stdout.isatty()

    def cmd_test(self, args: argparse.Namespace) -> int:
        kind, n = self.suite.targets()[args.target]

        # --dry-run promises to touch nothing, so it precedes every other step.
        if args.dry_run:
            backend = getattr(args, "backend", None) or self.env.detect_backend() or "?"
            for k, case in self.suite.collect_runs(kind, n):
                print(f"would run: {k} {case} (backend={backend})")
            return 0

        backend = self.env.require_backend(getattr(args, "backend", None))
        runs = self.suite.collect_runs(kind, n)

        problem = self.env.preflight(backend)
        if problem:
            print(f"error: {problem}", file=sys.stderr)
            return 1

        color = self._use_color(args.no_color)
        green = "\033[32m" if color else ""
        red = "\033[31m" if color else ""
        reset = "\033[0m" if color else ""

        results: list[tuple[str, str, int, float]] = []
        for k, case in runs:
            print(f"\n=== {k} test {case} (backend={backend}) ===")
            started = time.monotonic()
            try:
                rc = self.suite.run_case(k, case, backend, args.timeout)
            except ModelSourceUnavailable as e:
                print(f"skip: {e}", file=sys.stderr)
                rc = 2
            results.append((k, case, rc, time.monotonic() - started))
            if rc != 0 and args.fail_fast:
                break

        # Summary
        print("\n=== summary ===")
        worst = 0
        for k, case, rc, secs in results:
            status = f"{green}PASS{reset}" if rc == 0 else f"{red}FAIL (rc={rc}){reset}"
            print(f"  {k} {case}: {status}  ({secs:.1f}s)")
            worst = max(worst, rc)
        passed = sum(1 for r in results if r[2] == 0)
        total = sum(r[3] for r in results)
        print(f"{passed}/{len(results)} passed in {total:.1f}s")
        return worst

    # -- argparse -----------------------------------------------------------

    def common_parser(self) -> argparse.ArgumentParser:
        """Options accepted both before and after the subcommand."""
        c = argparse.ArgumentParser(add_help=False)
        c.add_argument(
            "--venv",
            metavar="PATH",
            default=argparse.SUPPRESS,
            help="virtualenv to test against; `install` creates it if missing. Every "
            "subprocess runs this interpreter directly instead of `uv run`, so the "
            "installed wheel is what gets tested even from inside the source checkout.",
        )
        c.add_argument(
            "--models-dir",
            "--models_dir",
            metavar="PATH",
            dest="models_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding the GGUF/safetensors models (default: {self.paths.models_dir})",
        )
        shorthand = c.add_mutually_exclusive_group()
        for backend in self.env.BACKENDS:
            shorthand.add_argument(
                f"--{backend}",
                dest="backend",
                action="store_const",
                const=backend,
                default=argparse.SUPPRESS,
                help=f"test the {backend} backend, in {self.venv_prefix}{backend} unless --venv says otherwise",
            )
        c.add_argument(
            "--data-dir",
            "--data_dir",
            metavar="PATH",
            dest="data_dir",
            default=argparse.SUPPRESS,
            help=f"directory holding corpus1.txt / jfk.wav (default: {self.paths.data_dir})",
        )
        return c

    @staticmethod
    def install_parser() -> argparse.ArgumentParser:
        """Options that only mean something while writing to the venv."""
        i = argparse.ArgumentParser(add_help=False)
        i.add_argument(
            "--wheel",
            metavar="WHEEL|SPEC",
            default=None,
            help="override what to install: a local wheel "
            "(dist/cyllama_cuda12-0.4.3-cp312-abi3-win_amd64.whl) or a pinned "
            "requirement (cyllama-vulkan==0.4.3). Usually unnecessary -- without "
            "it the latest release of the backend's distribution is fetched from "
            "the index (--cuda -> cyllama-cuda12).",
        )
        i.add_argument(
            "--with",
            dest="extra",
            action="append",
            metavar="PKG",
            help="extra package to install alongside the wheel (repeatable), "
            "e.g. --with numpy when diagnosing a wheel that predates a fix.",
        )
        i.add_argument(
            "--python",
            metavar="VERSION",
            help="interpreter for a venv created here (e.g. 3.12); passed to `uv venv --python`",
        )
        i.add_argument(
            "--upgrade",
            action="store_true",
            help="pass --upgrade to uv pip install",
        )
        i.add_argument(
            "--reinstall",
            action="store_true",
            help="pass --reinstall to uv pip install",
        )
        return i

    def build_parser(self) -> argparse.ArgumentParser:
        common = self.common_parser()
        p = argparse.ArgumentParser(
            description="cyllama wheel tester",
            parents=[common],
            epilog=("example: rwt.py install --cuda && rwt.py test --cuda test-all --models-dir models"),
        )
        _sub = p.add_subparsers(dest="cmd", required=True, metavar="<command>")

        class sub:  # noqa: N801 - thin shim so add_parser always inherits `common`
            @staticmethod
            def add_parser(
                name: str,
                parents: Sequence[argparse.ArgumentParser] = (),
                **kw: Any,
            ) -> argparse.ArgumentParser:
                return _sub.add_parser(name, parents=[common, *parents], **kw)

        sub.add_parser("info", help="show python/backend/models info").set_defaults(func=self.cmd_info)
        sub.add_parser("sync", help="uv sync project dependencies").set_defaults(func=self.cmd_sync)
        sub.add_parser("clean", help="remove the .venv directory").set_defaults(func=self.cmd_clean)
        sub.add_parser("reset", help="clean + sync").set_defaults(func=self.cmd_reset)

        inst = sub.add_parser(
            "install",
            parents=[self.install_parser()],
            help="install a cyllama wheel into --venv, creating it if needed",
        )
        inst.set_defaults(func=self.cmd_install)

        dl = sub.add_parser("download", help="download a model (or 'all')")
        dl.add_argument("key", choices=[*self.models.sources.keys(), "all"])
        dl.set_defaults(func=self.cmd_download)

        lst = sub.add_parser("list", help="list test targets and models (or one of them)")
        lst.add_argument(
            "what",
            nargs="?",
            choices=["tests", "models", "all"],
            default="all",
            help="which registry to show (default: both)",
        )
        lst.set_defaults(func=self.cmd_list)

        # The flat names this script used before `list` existed. Kept working, but
        # out of --help so there is one obvious spelling.
        sub.add_parser("list-models").set_defaults(func=self.cmd_list_models)
        sub.add_parser("list-tests").set_defaults(func=self.cmd_list_tests)

        gm = sub.add_parser("gen-makefile", help="generate the Makefile from this script's registries")
        gm.add_argument("-o", "--output", help="write to file instead of stdout (e.g. -o Makefile)")
        gm.set_defaults(func=self.cmd_gen_makefile)

        # `test` takes one target name -- `test-sd-3` rather than `test sd 3`, so a
        # target is a single token and matches the Makefile rule of the same name.
        t = sub.add_parser("test", help="run a test target (see `list tests`)")
        t.add_argument(
            "target",
            choices=list(self.suite.targets()),
            metavar="TARGET",
            help="one of the targets `list tests` prints, e.g. test-all, test-gen-1",
        )
        t.add_argument(
            "--timeout",
            type=float,
            default=None,
            help="per-test timeout in seconds (default: no timeout)",
        )
        t.add_argument(
            "--fail-fast",
            action="store_true",
            help="stop at the first failing test instead of running the full matrix",
        )
        t.add_argument(
            "--dry-run",
            action="store_true",
            help="print the test matrix without downloading or invoking anything",
        )
        t.add_argument(
            "--no-color",
            action="store_true",
            help="disable colored PASS/FAIL output in the summary",
        )

        t.set_defaults(func=self.cmd_test)

        return p

    def main(self, argv: list[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        self.configure(args)
        return int(args.func(args) or 0)


def main() -> None:
    sys.exit(Cli().main())


if __name__ == "__main__":
    main()
