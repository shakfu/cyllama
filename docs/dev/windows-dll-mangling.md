# Windows DLL name mangling: `--include` bypasses delvewheel's import rewrite

## Summary

Both published GPU wheels for Windows at 0.4.2 — `cyllama-cuda12` and
`cyllama-vulkan` — silently never load their GPU backend. The bundled plugin
(`ggml-cuda.dll` / `ggml-vulkan.dll`) is unloadable: delvewheel mangled
`ggml-base.dll` to `ggml-base-<hash>.dll` and rewrote every importer's PE import
table to match — except `ggml-cuda.dll`, which is pulled in with `--include` and
therefore is not part of the dependency graph delvewheel rewrites. It still
imports the pre-mangling name, which no longer exists in the wheel.

Result: `cyllama info` prints `built: CUDA` while `registries:` lists only `CPU`.
The package installs, imports, and passes its test suite, entirely on CPU.

`windows-cuda.md` predicted exactly this failure and prescribed the fix
(`--no-mangle`); the prescription was never applied. It is now, and a CUDA wheel
built with it reports `registries: CPU, CUDA` on a clean install -- see
Verification.

## Symptoms

The only user-visible signal is a single line with an empty error string:

```
load_backend: loaded CPU backend from ...\cyllama_cuda12.libs\ggml-cpu-<hash>.dll
load_backend: failed to load ...\cyllama_cuda12.libs\ggml-cuda.dll:
```

and then:

```
$ cyllama info
  built:         CUDA
  registries:    CPU
  devices:
    CPU                  [CPU  ]  AMD Ryzen 9 7940HX with Radeon Graphics
```

Workloads run at CPU speed. Measured on an RTX 4060, `run_wheel_test.py test gen all`:

| test | as shipped | after repair |
|---|---|---|
| gen 1 — Llama-3.2-1B | — | 155.8 tok/s |
| gen 2 — Qwen3-4B streamed | 10.6 tok/s | 53.1 tok/s |
| gen 3 — Gemma-4-E4B streamed | 12.2 tok/s | 7.3 tok/s |

All three pass in both configurations. Nothing errors; the wheel is simply slow.
(gen 3 being *slower* on GPU is a separate, unrelated issue — likely partial
offload thrashing an 8 GiB card.)

## Root cause

`manage.py:2442-2453` builds the Windows repair command from
`_WIN_INCLUDES` / `_WIN_EXCLUDES` (`manage.py:2211-2217`):

```
delvewheel repair -w {dest} --add-path <thirdparty dynamic dirs>
                  --include ggml-cuda.dll
                  --no-dll nvcuda.dll --no-dll cudart64_12.dll
                  --no-dll cublas64_12.dll --no-dll cublasLt64_12.dll
                  {wheel}
```

There is no `--no-mangle`.

ggml's backend plugins are loaded at runtime via `LoadLibraryW`, so nothing in
the wheel imports `ggml-cuda.dll` by name. That is why `--include` is needed at
all (see `packaging.md`). But the two mechanisms do not compose:

- delvewheel mangles bundled non-system DLLs to hash-suffixed names, then
  patches the import tables of everything **in the dependency graph** to match.
- `--include` copies a file into `.libs` **verbatim**. It is not in the graph —
  nothing imports it — so its own import table is never rewritten.

`ggml-cuda.dll` therefore keeps its build-time import of `ggml-base.dll`, a name
that no longer exists after the repair. The Windows loader cannot resolve it and
`LoadLibraryW` fails with no useful error.

## Confirmation

Independent of any runtime, from the published artifact alone. Download
`cyllama_cuda12-0.4.2-cp312-abi3-win_amd64.whl`
(sha256 `65131380f9dd0ce8405c920facda162fb987ac1bf2ab8e3cd9992b252d531672`),
unzip, and parse the PE import tables:

```python
import glob, os, pefile
L = "cyllama_cuda12.libs"
shipped = {f.lower() for f in os.listdir(L)}
for f in sorted(glob.glob(L + "/*.dll")) + sorted(glob.glob("cyllama/*/*.pyd")):
    pe = pefile.PE(f, fast_load=True)
    pe.parse_data_directories(
        directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    imps = [e.dll.decode().lower() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])]
    missing = [i for i in imps
               if i.startswith(("ggml", "llama", "mtmd")) and i not in shipped]
    if missing:
        print(f"{os.path.basename(f)}: MISSING {missing}")
    pe.close()
```

Output:

```
ggml-<hash>.dll        imports ['ggml-base-b034757f8f3c9df6e6af5e38d8541643.dll']
ggml-cpu-<hash>.dll    imports ['ggml-base-b034757f8f3c9df6e6af5e38d8541643.dll']
llama-<hash>.dll       imports ['ggml-<hash>.dll', 'ggml-base-b034757f8f3c9df6e6af5e38d8541643.dll']
mtmd-<hash>.dll        imports [..., 'ggml-base-b034757f8f3c9df6e6af5e38d8541643.dll']
llama_cpp.pyd          imports [..., 'ggml-base-b034757f8f3c9df6e6af5e38d8541643.dll']
stable_diffusion.pyd   imports [..., 'ggml-base-b034757f8f3c9df6e6af5e38d8541643.dll']
whisper_cpp.pyd        imports [..., 'ggml-base-b034757f8f3c9df6e6af5e38d8541643.dll']

ggml-cuda.dll          imports ['ggml-base.dll']   <-- MISSING FROM WHEEL
```

Seven files rewritten, one untouched. `ggml-cuda.dll` is also the only unmangled
*filename* in `cyllama_cuda12.libs` — a quick visual tell that a file skipped the
repair graph.

## Fix

Applied in `manage.py`. A `--no-mangle` list beside the existing include/exclude
tables:

```python
_WIN_NO_MANGLE: list[str] = [
    "ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "llama.dll", "mtmd.dll",
]
```

and one flag in the Windows branch of `_run_wheel_repair`, `;`-joined the way
delvewheel expects (`--no-mangle DLLS`, "DLL name(s) not to mangle,
';'-delimited"):

```python
if _WIN_NO_MANGLE:
    cmd += ["--no-mangle", ";".join(_WIN_NO_MANGLE)]
```

No narrower option exists: `--analyze-existing` vendors dependencies of DLLs the
*build* placed in the wheel, not of `--include`d ones, so nothing makes
delvewheel rewrite an `--include`d file's own import table.

Pinning only `ggml-base.dll` is sufficient for today's failure, but the whole
set is cheaper than rediscovering this per-plugin. Mangling exists to avoid
collisions with same-named DLLs from other wheels in the same process; these
five names are specific enough that the collision risk is negligible against a
backend that silently does not load.

Applied unconditionally rather than per-backend: the CPU wheel bundles none of
these libs (see the audit below), so the flag is a no-op there, and one shared
list means a future `--include`d plugin cannot reintroduce the bug.

## Audit of the published 0.4.2 Windows wheels

Every published Windows wheel, checked with the script above:

| wheel | `.libs` contents | result |
|---|---|---|
| `cyllama_cuda12-0.4.2-cp312-abi3-win_amd64` | mangled ggml/llama/mtmd + **unmangled `ggml-cuda.dll`** | **BROKEN** — `ggml-cuda.dll` imports missing `ggml-base.dll` |
| `cyllama_vulkan-0.4.2-cp312-abi3-win_amd64` | mangled ggml/llama/mtmd + **unmangled `ggml-vulkan.dll`** | **BROKEN** — `ggml-vulkan.dll` imports missing `ggml-base.dll` |
| `cyllama-0.4.2-cp312-abi3-win_amd64` (CPU) | only `msvcp140`, `msvcp140_codecvt_ids`, `vcomp140` | clean |

**Both GPU wheels are affected, identically.** The vulkan job uses the same
`--include` pattern (`_WIN_INCLUDES["vulkan"] = ["ggml-vulkan.dll"]`) with no
`--no-mangle`, and its `ggml-vulkan.dll` is likewise the sole unmangled filename
in its `.libs`, still importing `ggml-base.dll`. Confirmed by PE parse, not
inferred from the shared code path.

The CPU wheel is unaffected for a structural reason, not by luck: it ships no
ggml DLLs at all — ggml is static-linked into the `.pyd` extensions — so there is
nothing to mangle and nothing loaded by `LoadLibraryW`. That also means the CPU
wheel cannot regress this way, and that any backend gaining a dynamically-loaded
plugin inherits the bug.

No `cyllama-rocm` or `cyllama-sycl` distributions exist on PyPI, so there is
nothing further to check; `_WIN_INCLUDES` has no entry for them either.

## Verification: a CUDA wheel built with the fix

Built locally on the box described above:

```
CMAKE_CUDA_ARCHITECTURES=89 python scripts/manage.py wheel_build \
    --backend cuda --dynamic --abi3
```

`89` is this card's native compute capability rather than the `75` CI pins; the
packaging fix is architecture-independent, so this only makes the artifact local
rather than a drop-in for the published one. Note that `wheel_build` hardcodes
`SD_USE_VENDORED_GGML=0` for dynamic non-CPU backends, so it takes the
from-source nvcc path rather than the `download_release()` fast path
`windows-cuda.md` describes -- slow, but it is what the Makefile's
`build-cuda-dynamic` does, so the artifact matches a CI-built one.

The repair now emits:

```
delvewheel repair -w dist --add-path thirdparty\llama.cpp\dynamic
                  --include ggml-cuda.dll
                  --no-dll nvcuda.dll --no-dll cudart64_12.dll
                  --no-dll cublas64_12.dll --no-dll cublasLt64_12.dll
                  --no-mangle ggml.dll;ggml-base.dll;ggml-cpu.dll;llama.dll;mtmd.dll
                  dist\cyllama-0.4.2-cp312-abi3-win_amd64.whl
```

and the resulting wheel parses clean under the script above:

```
shipped: ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll, ggml.dll, llama.dll,
         msvcp140-<hash>.dll, msvcp140_codecvt_ids-<hash>.dll, mtmd.dll,
         vcomp140-<hash>.dll
=> 0 file(s) with unresolvable in-wheel imports
```

The five project libs are unmangled; the MSVC runtimes keep their hashes, which
is the point of scoping the list rather than reaching for `--no-mangle-all` --
cross-wheel collision protection still applies everywhere it matters.

Installed into a clean venv, with no patching of any kind:

| check | result |
|---|---|
| `cyllama info` | `built: CUDA`, `registries: CPU, CUDA`, `CUDA0 [GPU] RTX 4060` |
| `run_wheel_test.py test gen all` | 3/3 PASS -- 163.1 / 53.3 / 57.8 tok/s |
| `run_wheel_test.py test sd 3` | PASS -- 40.66s, sampling 37.2s on CUDA0 |
| `cyllama embed` | PASS -- also the first real-install test of the backend-load fix below |

Two caveats on those numbers. `run_wheel_test.py` labels the runs `backend=cpu`
because `detect_backend()` matches on the `cyllama-cuda12` distribution name and
a local build is plain `cyllama`; the label is wrong, the execution is CUDA.
And gen 3 ran at 57.8 tok/s here against 7.3 on the byte-patched published wheel
-- unexplained, and this build differs in more than one way (sm_89 vs sm_75,
dynamic vs the published config), so the earlier gen-3 slowness should be treated
as still open rather than fixed.

### `wheel_build --dynamic` could not reach the repair step

Building the above first required fixing an unrelated crash. `do_wheel_build`
calls `do_wheel_repair` directly with a hand-built namespace instead of going
through the parser:

```python
self.do_wheel_repair(argparse.Namespace(backend=args.backend, wheel=None, dest_dir=None))
```

`do_wheel_repair` reads `args.archs` unconditionally, so every dynamic build on
every platform ended in `AttributeError: 'Namespace' object has no attribute
'archs'`. It fires *after* `uv build` succeeds, which is what made it survive: it
leaves a built-but-unrepaired wheel in `dist/` -- no bundled libs, no `.libs`
directory -- and looks like a successful build to anyone who checks the artifact
rather than the traceback. The namespace now passes `archs=None`.

One environment snag, not a bug: `_run_wheel_repair` calls
`_install("delvewheel")`, which shells out to `pip`. A uv-created `.venv` has no
pip, so a local repair fails there until pip is installed. CI is unaffected --
cibuildwheel environments have pip.

## Unrelated but adjacent: CUDA runtime DLLs are not bundled

`cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll` are `--no-dll`
excluded by design — users supply the CUDA runtime. That is a deliberate choice
(`windows-cuda.md`), not part of this bug, but it means fixing the mangling
alone does not guarantee the backend loads on a machine without a CUDA toolkit
on `PATH`. Worth testing separately; the verification above was done on a box
with CUDA 12.9 installed. See also the `CHANGELOG` note about the
`cudart-llama-bin-win-cuda-*` companion asset, which was intended to address
this.

## Why CI did not catch it

The guardrail already exists and is correct.
`build-gpu-wheels-abi3.yml:833-853` installs the wheel into a clean venv and
`ctypes.WinDLL()`-loads every `cyllama*.libs/ggml-*.dll` using the same
`LoadLibraryW(path, NULL, 0)` semantics ggml uses in C. That is precisely the
call that fails here, and it fails on a GPU-less runner too — no GPU or driver
needed, since the failure is an unresolvable import, not a missing device.

So the check would have caught this. What remains unverified is whether it ran:
the whole smoke job is gated on `steps.check.outputs.skip`, derived from
`inputs.cuda_windows`, and is skipped entirely when that input is false. The
other Windows CUDA path, `build-new-wheels.yml:108`, uses a bare
`delvewheel repair` with no `--include` at all — it cannot be the source, since
the shipped wheel does contain `ggml-cuda.dll`. Checking the Actions log for the
0.4.2 `cuda-windows` build would settle which run produced the artifact and
whether the smoke job was skipped.

Recommended follow-up regardless: make the DLL link-test a release gate rather
than an input-gated matrix leg, so a wheel cannot be published without it.

## Local repair (diagnosis only)

To verify a fix hypothesis against an already-installed wheel without a rebuild,
rename the mangled file and byte-patch the import tables. PE import names are
null-terminated strings, so replacing a longer name with a shorter one padded
back to the original length keeps every offset valid:

```python
import glob, os, sys
SP = sys.argv[1]                      # .../site-packages
L = os.path.join(SP, "cyllama_cuda12.libs")
m = glob.glob(os.path.join(L, "ggml-base-*.dll"))
if m:
    os.rename(m[0], os.path.join(L, "ggml-base.dll"))
    old = os.path.basename(m[0]).encode() + b"\x00"
    new = b"ggml-base.dll\x00".ljust(len(old), b"\x00")
    for t in (glob.glob(os.path.join(L, "*.dll"))
              + glob.glob(os.path.join(SP, "cyllama", "*", "*.pyd"))):
        d = open(t, "rb").read()
        if old in d:
            open(t, "wb").write(d.replace(old, new))
```

Everyone moves onto `ggml-cuda.dll`'s spelling rather than the reverse, because
a 46-byte name can shrink in place but a 14-byte one cannot grow. Afterwards
`cyllama info` reports `registries: CPU, CUDA`.

**Do not ship this**, and do not attempt the tempting shortcut of leaving both
filenames in place as separate copies. Two files means two `ggml-base` module
instances with two copies of ggml's static state; the process crashes at exit
(observed as exit code 127) once the CUDA backend actually registers.

## Separate bug found alongside this one (fixed)

Four entry points constructed a `LlamaModel` with no preceding
`ggml_backend_load_all()`, so llama.cpp refused to load into an empty registry:

```
$ cyllama embed -m models/bge-small-en-v1.5-q8_0.gguf -f tests/media/corpus1.txt ...
llama_model_load_from_file_impl: no backends are loaded. hint: use ggml_backend_load()
ValueError: Failed to load model from file: ...
```

`rag/embedder.py`, `rag/advanced.py` (`Reranker._ensure_model`), `memory.py`
(`dump_metadata_json`) and `llama/server/python.py` (`Server.load_model`) were
missing it; `llama/cli.py:354`, `llama/chat.py:95`, `sd/__init__.py:111`,
`llama/tts.py:153`, `whisper/cli.py:380` and `batching.py:129` all had it. The
registry is process-global, so the three library entry points failed only when
nothing else had already loaded backends — which is why this survived: in a
normal RAG flow the `Embedder` runs first and the `Reranker` inherits its
registry.

`dump_metadata_json()` deserves separate mention. It catches the load failure
and falls back to `_DEFAULT_METADATA`, so it did not raise — it returned another
model's shape. For `bge-small-en-v1.5-q8_0.gguf` it reported
`arch=llama, block_count=32, embedding_length=4096` against a true
`arch=bert, block_count=12, embedding_length=384`, and `cyllama memory` sized its
KV-cache estimate off those numbers. A wrong answer with no error, which is worse
than the hard failure the other three gave.

Backend-independent — this affected the plain `cyllama` CPU wheel too, and has
nothing to do with the mangling bug above beyond being found while chasing it.
