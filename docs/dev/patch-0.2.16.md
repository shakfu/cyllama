# cyllama 0.2.16 — SD "No devices found!" patch

## Symptom

On 0.2.16 (any backend variant — `cyllama-cuda12`, `cyllama-vulkan`, etc.),
constructing an `SDContext` fails to register the GPU backend:

```
[ERROR] util.cpp:711  - No devices found!
[WARN]  util.cpp:752  - loading CPU backend
...
GGML_ASSERT(backend) failed   (ggml-backend.cpp:238)
```

0.2.15 is unaffected.

## Cause

stable-diffusion.cpp `master-592-b8079e2` (PR #1448) switched from
compile-time to runtime backend discovery. cyllama 0.2.16 bumped sd.cpp to
`master-593-3d6064b` but the Python wrapper still assumes the old static
auto-registration. `ggml_backend_load_all()` is exposed but never called,
so when `new_sd_ctx` runs no backend is registered → CPU fallback → null
backend abort in `ggml_backend_alloc_ctx_tensors`.

## Option A — runtime workaround (no patching)

Call the loader before constructing any `SDContext`:

```python
import cyllama.sd as sd

sd.ggml_backend_load_all()      # register GPU backends
ctx = sd.SDContext(params)
```

## Option B — patch the installed package

Edit `<site-packages>/cyllama/sd/__init__.py`. Locate the import block
that ends with `set_preview_callback,` and the `__all__ = [` line that
follows, then insert a call to `ggml_backend_load_all()` between them:

```python
    set_preview_callback,
)

# sd.cpp master-592 (#1448) switched to runtime backend discovery; register
# backends at import time so GPU support works without explicit user action.
ggml_backend_load_all()

__all__ = [
```

Then clear the cached bytecode so the edit takes effect:

```bash
rm -f <site-packages>/cyllama/sd/__pycache__/__init__.cpython-*.pyc
```

Locate `<site-packages>` with:

```bash
python -c "import cyllama, os; print(os.path.dirname(cyllama.__file__))"
```

## Verification

```bash
python -c "import cyllama.sd as sd; print(sd.get_system_info())"
```

Expected output includes a `load_backend: loaded CUDA backend from ...`
(or Vulkan/Metal) line and your GPU listed with its compute capability.

## Upstream fix

This should be fixed in cyllama itself by calling `ggml_backend_load_all()`
at the bottom of `src/cyllama/sd/__init__.py` after the import block, so
import-time behavior matches the pre-592 static-link build. Until a
release ships that fix, use Option A or Option B.
