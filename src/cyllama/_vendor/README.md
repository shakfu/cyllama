# cyllama vendored dependencies

This directory contains pure-Python copies of upstream libraries that
cyllama depends on internally. They are loaded under the
`cyllama._vendor.*` namespace so they do not collide with whatever
versions the user has installed in their own environment.

## Why vendor?

cyllama is distributed as binary wheels with bundled native libraries
(libllama, libggml, libwhisper, libsd — easily 30-90 MB depending on
the GPU backend). The package philosophy is "everything you need is in
the wheel." Adding ~600 KB of vendored Python to fit that model is a
rounding error, and the alternative — declaring the deps in
`pyproject.toml` and letting pip resolve them — introduces transitive
dependencies that can conflict with the user's environment, and
weakens reproducibility (different `jinja2` versions can render the
same chat template differently).

The trade-off is that security patches for vendored libraries require
a cyllama release rather than a `pip install --upgrade`. Mitigating
factors: the libraries we vendor (currently `jinja2` and `markupsafe`)
have very short security histories, and cyllama uses them only to
evaluate trusted GGUF chat templates inside `ImmutableSandboxedEnvironment`,
which is the safe-evaluation mode designed precisely for untrusted
input.

## Currently vendored

| Library | Version | License | Purpose |
|---|---|---|---|
| `jinja2` | 3.1.6 | BSD-3-Clause | Evaluating GGUF chat templates |
| `markupsafe` | 3.0.3 | BSD-3-Clause | Required by `jinja2` |

License files are at `LICENSE.jinja2` and `LICENSE.markupsafe` in this
directory.

`markupsafe` ships an optional `_speedups.c` C extension for
HTML-escaping performance. cyllama does not vendor it because (a) we
never HTML-escape model output, so the slow path is functionally
indistinguishable from the fast path here, and (b) vendoring the C
extension would reintroduce build complexity that pure-Python
vendoring is meant to avoid. `markupsafe` falls back to its
`_native.py` pure-Python implementation automatically when
`_speedups` is missing.

## Import path rewriting

Two rewrites are applied to the vendored `jinja2` source after copying
from upstream. Both are done by `scripts/vendor_jinja2.sh` and must be
re-applied on every re-vendor. The rewrites are idempotent and visible
in the diff so reviewers can verify nothing else was touched.

### 1. `markupsafe` static imports

`jinja2` does `from markupsafe import ...` and `import markupsafe`
across 9 of its source files. Without rewriting, these would resolve
to whatever `markupsafe` is on the user's `sys.path` (or fail with
`ImportError` if none is installed). The rewrite changes them to
`from cyllama._vendor.markupsafe import` /
`import cyllama._vendor.markupsafe as markupsafe`.

### 2. `compiler.py` runtime emit string

`jinja2`'s `compiler.py` emits a literal `from jinja2.runtime import ...`
line into the Python source of every *compiled template*. Without
rewriting this emit string, every compiled template would try to
import from the top-level `jinja2.runtime` module, which means:

* If the user has `jinja2` installed in their environment, compiled
  templates from the vendored `jinja2` would pull runtime symbols from
  the *user's* `jinja2`, mixing two versions in undefined ways. The
  observable failure mode is `_MissingType` sentinels leaking into
  rendered output instead of being converted to `Undefined`, because
  the cross-module identity check `if rv is missing:` fails between
  the two `missing` sentinels.
* If the user has no `jinja2` installed, the import fails entirely
  and template rendering raises `ImportError`.

The rewrite changes the emit string to
`from cyllama._vendor.jinja2.runtime import ...` so compiled templates
always pull from the vendored runtime. This is the same trick `pip._vendor`
uses to keep its vendored copies hermetic.

The same pattern is used by `pip._vendor` and `setuptools._vendor`.

## Do not modify

Files under `jinja2/` and `markupsafe/` are read-only — any local
modifications will be lost on the next re-vendor. If a vendored
library needs a fix:

1. File the fix upstream (https://github.com/pallets/jinja or
   https://github.com/pallets/markupsafe).
2. Wait for it to ship in a release.
3. Re-run `scripts/vendor_jinja2.sh` to pull in the new version.
4. Verify cyllama's tests still pass.
5. Commit the updated vendor directory and bump this README.

## Re-vendoring

To update the vendored libraries to a newer version:

```bash
./scripts/vendor_jinja2.sh [JINJA2_VERSION] [MARKUPSAFE_VERSION]
```

Defaults to the latest stable releases. The script handles downloading,
extracting, copying, removing the C extension, and re-applying the
import path rewrites. After running it, update the version table above.
