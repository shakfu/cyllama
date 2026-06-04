"""Translate opaque native-library load failures into actionable errors.

Some wheels deliberately do not vendor a backend's userspace runtime -- the
SYCL wheel omits the Intel oneAPI runtimes (see installation docs). When that
runtime is absent from the loader path, importing cyllama fails with a bare
``libsycl.so.8: cannot open shared object file``, which doesn't tell the user
what to install.

This module rewrites such failures using the per-backend host-runtime soname
lists recorded in ``build_config.json`` (emitted by ``manage.py`` from the
wheel-repair exclude lists -- the single source of truth, so there is no second
copy of the soname list to keep in sync). Only the human-facing remediation
prose lives here; the detection data is build-generated.
"""

from __future__ import annotations

from . import build_config as _bc

# Per-backend remediation text, keyed by backend name. A backend appears here
# only if its wheel intentionally ships without its runtime. The detection data
# (which sonames signal the missing runtime) is read from build_config, not
# duplicated here.
_REMEDIATION: dict[str, str] = {
    "sycl": (
        "cyllama-sycl could not load a required Intel oneAPI runtime "
        "library:\n    {msg}\n\n"
        "The SYCL wheel does not vendor the oneAPI userspace runtimes "
        "(libsycl, MKL, TBB, libiomp5, the Intel compiler runtimes). "
        "Install them and put them on the loader path before importing "
        "cyllama. On Debian/Ubuntu, after adding the Intel oneAPI APT "
        "repo:\n\n"
        "    sudo apt install \\\n"
        "        intel-oneapi-runtime-dpcpp-cpp \\\n"
        "        intel-oneapi-runtime-mkl \\\n"
        "        intel-oneapi-runtime-tbb \\\n"
        "        intel-oneapi-runtime-openmp\n"
        "    source /opt/intel/oneapi/setvars.sh\n\n"
        "See docs/installation.md#cyllama-sycl-host-prerequisites for "
        "RPM-based distros, the separate GPU/CPU device-driver "
        "requirement, and links to Intel's install guides."
    ),
}


def _soname_in_message(soname: str, msg: str) -> bool:
    """Tolerant substring test of a (possibly glob) soname against *msg*.

    ``build_config`` stores the wheel-repair patterns verbatim, a few of which
    are globs (e.g. ``libmkl_*.so*``). Reduce each to the literal stem before
    the first glob character and test that against the loader's message.
    """
    stem = soname.split("*", 1)[0]
    return bool(stem) and stem in msg


def translate(exc: BaseException) -> None:
    """Raise an actionable ``ImportError`` if *exc* is a missing host runtime.

    Returns normally -- so the caller re-raises the original exception -- when
    the failure isn't a recognised unvendored-runtime case.
    """
    msg = str(exc)
    for name, remediation in _REMEDIATION.items():
        if not _bc.backend_enabled(name):
            continue
        if any(_soname_in_message(so, msg) for so in _bc.host_runtimes(name)):
            raise ImportError(remediation.format(msg=msg)) from exc
