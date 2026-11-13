"""Tests for `check_sqlite_vector_vendor` in `scripts/manage.py`."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

MANAGE_PY = Path(__file__).resolve().parent.parent / "scripts" / "manage.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


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
    (up / "src" / "sv.c").write_text(UPSTREAM_C)
    (up / "src" / "sv.h").write_text("#define V 1\n")
    (up / "libs" / "sqlite3.h").write_text("/* sqlite */\n")
    (up / "libs" / "fp16" / "fp16.h").write_text("/* fp16 */\n")
    (up / "LICENSE.md").write_text("Apache-2.0\n")

    ven = tmp_path / "ven"
    (ven / "fp16").mkdir(parents=True)
    (ven / "sv.c").write_text(PATCHED_C)
    (ven / "sv.h").write_text("#define V 1\n")
    (ven / "sqlite3.h").write_text("/* sqlite */\n")
    (ven / "fp16" / "fp16.h").write_text("/* fp16 */\n")
    (ven / "LICENSE.md").write_text("Apache-2.0\n")

    patch = tmp_path / "sqlite-vector-test.patch"
    patch.write_text(PATCH)
    return ven, up, patch


def test_patched_tree_matches(trees):
    ven, up, patch = trees
    assert check(ven, up, [patch]) == []


def test_missing_patch_reported(trees):
    ven, up, _ = trees
    assert check(ven, up, []) == ["sv.c: differs from upstream plus patches"]


def test_change_beside_patched_hunk(trees):
    ven, up, patch = trees
    (ven / "sv.c").write_text(PATCHED_C.replace("return 0", "return 1"))
    assert check(ven, up, [patch]) == ["sv.c: differs from upstream plus patches"]


def test_upstream_change_under_old_pin(trees):
    ven, up, patch = trees
    (up / "src" / "sv.c").write_text(UPSTREAM_C + "int g(void);\n")
    assert check(ven, up, [patch]) == ["sv.c: differs from upstream plus patches"]


def test_patch_that_no_longer_applies(trees):
    ven, up, patch = trees
    (up / "src" / "sv.c").write_text("int h(void);\n")
    (problem,) = check(ven, up, [patch])
    assert problem.startswith("sqlite-vector-test.patch: does not apply")


def test_unpatched_file_must_match(trees):
    ven, up, patch = trees
    (ven / "sv.h").write_text("#define V 2\n")
    assert check(ven, up, [patch]) == ["sv.h: differs from upstream plus patches"]


def test_extra_vendored_file(trees):
    ven, up, patch = trees
    (ven / "extra.c").write_text("")
    assert check(ven, up, [patch]) == ["extra.c: no counterpart in sqlite-vector checkout"]


def test_new_upstream_source_not_vendored(trees):
    ven, up, patch = trees
    (up / "src" / "distance-new.c").write_text("")
    (up / "libs" / "fp16" / "bitcasts.h").write_text("")
    assert check(ven, up, [patch]) == [
        "src/distance-new.c: not vendored",
        "libs/fp16/bitcasts.h: not vendored",
    ]
