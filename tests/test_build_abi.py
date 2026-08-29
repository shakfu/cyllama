"""Guards on the ggml ABI that cyllama's build has to keep consistent.

stable-diffusion.cpp and llama.cpp are compiled separately but the shipped
extension links *one* ggml, so anything that changes `struct ggml_tensor`'s
layout has to be identical on both sides. `GGML_MAX_NAME` does: `name` is an
inline `char[GGML_MAX_NAME]` in the struct, and `extra` -- the field right
after it, and the last one -- moves with it. SD writes `tensor->extra` on
every graph-cut segment boundary, so a mismatch puts those writes on top of
the next `ggml_object` header. Nothing fails to compile or link; the arena is
simply corrupted at runtime.

The pin went stale once (128 while upstream had moved to 160), so these tests
check both directions: the value manage.py propagates to llama.cpp, and the
value cyllama's own CMakeLists compiles against.
"""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANAGE_PY = ROOT / "scripts" / "manage.py"


@pytest.fixture(scope="module")
def manage():
    """Import scripts/manage.py as a module (it is stdlib-only by design)."""
    spec = importlib.util.spec_from_file_location("cyllama_manage", MANAGE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cyllama_manage"] = module
    spec.loader.exec_module(module)
    return module


def test_cmakelists_matches_the_propagated_value(manage):
    """cyllama's own translation units must use the value manage.py ships."""
    pinned = manage.StableDiffusionCppBuilder.GGML_MAX_NAME
    text = (ROOT / "CMakeLists.txt").read_text()
    found = re.findall(r"add_definitions\(-DGGML_MAX_NAME=(\d+)\)", text)
    assert found, "cyllama's CMakeLists.txt no longer defines GGML_MAX_NAME"
    assert [int(v) for v in found] == [pinned] * len(found)


def test_verify_rejects_a_stale_pin(manage, tmp_path, monkeypatch):
    """A drifted upstream value must fail the build, not warn."""
    builder = manage.StableDiffusionCppBuilder()
    (tmp_path / "CMakeLists.txt").write_text("add_definitions(-DGGML_MAX_NAME=%d)\n" % (builder.GGML_MAX_NAME + 32))
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: tmp_path))
    with pytest.raises(RuntimeError, match="GGML_MAX_NAME"):
        builder._verify_ggml_max_name()


def test_verify_accepts_a_current_pin(manage, tmp_path, monkeypatch):
    builder = manage.StableDiffusionCppBuilder()
    (tmp_path / "CMakeLists.txt").write_text("add_definitions(-DGGML_MAX_NAME=%d)\n" % builder.GGML_MAX_NAME)
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: tmp_path))
    builder._verify_ggml_max_name()  # must not raise


def test_verify_is_skipped_for_a_vendored_ggml(manage, tmp_path, monkeypatch):
    """With SD on its own ggml there is no shared struct to keep in step."""
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "1")
    builder = manage.StableDiffusionCppBuilder()
    (tmp_path / "CMakeLists.txt").write_text("add_definitions(-DGGML_MAX_NAME=9999)\n")
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: tmp_path))
    builder._verify_ggml_max_name()  # must not raise
