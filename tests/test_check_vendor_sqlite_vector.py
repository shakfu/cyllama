"""Tests for `check_sqlite_vector_vendor` in `scripts/manage.py`."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

MANAGE_PY = Path(__file__).resolve().parent.parent / "scripts" / "manage.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _lf(text: str) -> bytes:
    # write_text would emit CRLF on Windows, which git apply rejects in the patch.
    return text.encode()


def _load_manage():
    """Import scripts/manage.py as a module (it is not on the package path)."""
    if "cyllama_manage" in sys.modules:
        return sys.modules["cyllama_manage"]
    spec = importlib.util.spec_from_file_location("cyllama_manage", MANAGE_PY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cyllama_manage"] = module
    spec.loader.exec_module(module)
    return module


check = _load_manage().check_sqlite_vector_vendor

UPSTREAM_C = "#include <a.h>\n#include <b.h>\n\nint f(void) { return 0; }\n"
PATCHED_C = "#include <a.h>\n#include <b.h>\n#define X 1\n\nint f(void) { return 0; }\n"
PATCH = """--- a/src/sv.c
+++ b/src/sv.c
@@ -1,4 +1,5 @@
 #include <a.h>
 #include <b.h>
+#define X 1

 int f(void) { return 0; }
"""


@pytest.fixture
def trees(tmp_path):
    up = tmp_path / "up"
    (up / "src").mkdir(parents=True)
    (up / "libs" / "fp16").mkdir(parents=True)
    (up / "src" / "sv.c").write_bytes(_lf(UPSTREAM_C))
    (up / "src" / "sv.h").write_bytes(_lf("#define V 1\n"))
    (up / "libs" / "sqlite3.h").write_bytes(_lf("/* sqlite */\n"))
    (up / "libs" / "fp16" / "fp16.h").write_bytes(_lf("/* fp16 */\n"))
    (up / "LICENSE.md").write_bytes(_lf("Apache-2.0\n"))

    ven = tmp_path / "ven"
    (ven / "fp16").mkdir(parents=True)
    (ven / "sv.c").write_bytes(_lf(PATCHED_C))
    (ven / "sv.h").write_bytes(_lf("#define V 1\n"))
    (ven / "sqlite3.h").write_bytes(_lf("/* sqlite */\n"))
    (ven / "fp16" / "fp16.h").write_bytes(_lf("/* fp16 */\n"))
    (ven / "LICENSE.md").write_bytes(_lf("Apache-2.0\n"))

    patch = tmp_path / "sqlite-vector-test.patch"
    patch.write_bytes(_lf(PATCH))
    return ven, up, patch


def test_patched_tree_matches(trees):
    ven, up, patch = trees
    assert check(ven, up, [patch]) == []


def test_missing_patch_reported(trees):
    ven, up, _ = trees
    assert check(ven, up, []) == ["sv.c: differs from upstream plus patches"]


def test_change_beside_patched_hunk(trees):
    ven, up, patch = trees
    (ven / "sv.c").write_bytes(_lf(PATCHED_C.replace("return 0", "return 1")))
    assert check(ven, up, [patch]) == ["sv.c: differs from upstream plus patches"]


def test_upstream_change_under_old_pin(trees):
    ven, up, patch = trees
    (up / "src" / "sv.c").write_bytes(_lf(UPSTREAM_C + "int g(void);\n"))
    assert check(ven, up, [patch]) == ["sv.c: differs from upstream plus patches"]


def test_patch_that_no_longer_applies(trees):
    ven, up, patch = trees
    (up / "src" / "sv.c").write_bytes(_lf("int h(void);\n"))
    (problem,) = check(ven, up, [patch])
    assert problem.startswith("sqlite-vector-test.patch: does not apply")


def test_unpatched_file_must_match(trees):
    ven, up, patch = trees
    (ven / "sv.h").write_bytes(_lf("#define V 2\n"))
    assert check(ven, up, [patch]) == ["sv.h: differs from upstream plus patches"]


def test_extra_vendored_file(trees):
    ven, up, patch = trees
    (ven / "extra.c").write_bytes(_lf(""))
    assert check(ven, up, [patch]) == ["extra.c: no counterpart in sqlite-vector checkout"]


def test_new_upstream_source_not_vendored(trees):
    ven, up, patch = trees
    (up / "src" / "distance-new.c").write_bytes(_lf(""))
    (up / "libs" / "fp16" / "bitcasts.h").write_bytes(_lf(""))
    assert check(ven, up, [patch]) == [
        "src/distance-new.c: not vendored",
        "libs/fp16/bitcasts.h: not vendored",
    ]


# Sources CMakeLists.txt compiles for MSVC x64. distance-neon.c / distance-rvv.c
# are arch-guarded and never reach MSVC x64.
MSVC_SOURCES = ["sqlite-vector.c", "distance-cpu.c", "distance-avx2.c", "distance-avx512.c", "distance-sse2.c"]


def _unguarded_builtins(text: str) -> list[int]:
    """Line numbers of `__builtin_*` calls not inside an `#if` on __GNUC__ / __clang__."""
    import re

    stack: list[list] = []  # [condition, in_first_branch]
    bad = []
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        m = re.match(r"#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", s)
        if m:
            kw, cond = m.groups()
            if kw in ("if", "ifdef", "ifndef"):
                stack.append([("!" if kw == "ifndef" else "") + cond, True])
            elif kw == "elif" and stack:
                stack[-1] = [cond, False]
            elif kw == "else" and stack:
                stack[-1][1] = False
            elif kw == "endif" and stack:
                stack.pop()
            continue
        if "__builtin_" in s.split("//")[0]:
            if not any(first and re.search(r"__GNUC__|__clang__", c) and not c.startswith("!") for c, first in stack):
                bad.append(n)
    return bad


@pytest.mark.parametrize("name", MSVC_SOURCES)
def test_vendored_sources_guard_gcc_builtins(name):
    # MSVC has no __builtin_*; an unguarded call links on gcc/clang and fails with
    # LNK2019 on Windows only. Upstream's distance-avx2.c does this; the patch guards it.
    text = (MANAGE_PY.parent.parent / "thirdparty" / "sqlite-vector" / name).read_text()
    assert _unguarded_builtins(text) == []


def test_builtin_scan_detects_unguarded_call():
    assert _unguarded_builtins("int f(int x) { return __builtin_popcount(x); }\n") == [1]
    guarded = "#if defined(__GNUC__) || defined(__clang__)\n__builtin_popcount(x);\n#else\nslow(x);\n#endif\n"
    assert _unguarded_builtins(guarded) == []
    assert _unguarded_builtins("#ifdef __GNUC__\n#else\n__builtin_popcount(x);\n#endif\n") == [3]
