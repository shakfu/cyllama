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
check every direction: the value manage.py propagates to llama.cpp, and the
value cyllama's own CMakeLists and xcframework script compile against.
"""

import importlib.util
import inspect
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


def test_every_tree_that_links_the_shared_ggml_gets_the_define(manage):
    """llama.cpp, whisper.cpp and stable-diffusion.cpp must all be built with it.

    cyllama's CMakeLists takes every ggml lib from ``${LLAMACPP_LIB}``, so all
    three trees' object code calls *llama.cpp's* ggml -- whisper's own
    ``libggml.a`` is built and installed but never linked. A tree left on the
    default 64 is compiled against a ``ggml_tensor`` 96 bytes shorter than the
    one being allocated.
    """
    # stable-diffusion.cpp is deliberately absent: it *sets* the value in its
    # own CMakeLists, and `_verify_ggml_max_name()` checks cyllama's pin against
    # it. These two have to follow.
    followers = [manage.LlamaCppBuilder(), manage.WhisperCppBuilder()]
    for builder in followers:
        # The value reaches cmake as a raw -D in CMAKE_{C,CXX}_FLAGS; assert the
        # builder is wired to emit it rather than re-deriving how.
        source = inspect.getsource(type(builder).build)
        assert "GGML_MAX_NAME" in source, (
            f"{builder.name} does not propagate GGML_MAX_NAME to its cmake configure; "
            f"it will compile against a different struct ggml_tensor than the ggml it links"
        )


def test_cmakelists_matches_the_propagated_value(manage):
    """cyllama's own translation units must use the value manage.py ships."""
    pinned = manage.StableDiffusionCppBuilder.GGML_MAX_NAME
    text = (ROOT / "CMakeLists.txt").read_text()
    found = re.findall(r"add_definitions\(-DGGML_MAX_NAME=(\d+)\)", text)
    assert found, "cyllama's CMakeLists.txt no longer defines GGML_MAX_NAME"
    assert [int(v) for v in found] == [pinned] * len(found)


def test_xcframework_matches_the_propagated_value(manage):
    """The xcframework build configures ggml itself and must agree too."""
    pinned = manage.StableDiffusionCppBuilder.GGML_MAX_NAME
    text = (ROOT / "scripts" / "make_xcframework.py").read_text()
    found = re.findall(r"-DGGML_MAX_NAME=(\d+)", text)
    assert found, "make_xcframework.py no longer sets GGML_MAX_NAME"
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


def _staged_swap(manage, tmp_path, monkeypatch):
    """A checked-out SD tree, a built cmake dir under it, and a newer llama ggml."""
    src = tmp_path / "src"
    sd_dir = src / "stable-diffusion.cpp"
    (src / "llama.cpp" / "ggml" / "src").mkdir(parents=True)
    (src / "llama.cpp" / "ggml" / "src" / "ggml-metal-device.m").write_text("// split per op-source\n")
    (sd_dir / "ggml" / "src").mkdir(parents=True)
    (sd_dir / "ggml" / "src" / "ggml-metal.m").write_text("// one library\n")
    (sd_dir / "build").mkdir()
    (sd_dir / "build" / "ggml-metal-device.m.o").write_text("stale object")

    builder = manage.StableDiffusionCppBuilder()
    monkeypatch.setattr(builder.project, "src", src)
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: sd_dir))
    return builder, sd_dir


def test_swapping_ggml_drops_the_build_tree(manage, tmp_path, monkeypatch):
    """Objects compiled against the replaced ggml must not survive the swap.

    ``copytree`` preserves mtimes, so the incoming sources are not newer than
    the objects already in the cmake tree and make relinks against them.
    """
    builder, sd_dir = _staged_swap(manage, tmp_path, monkeypatch)

    builder._sync_ggml_abi()

    assert (sd_dir / "ggml" / "src" / "ggml-metal-device.m").exists(), "swap did not happen"
    assert not (sd_dir / "build").exists(), "cmake tree survived a ggml swap; its objects are stale"


def test_a_skipped_swap_leaves_the_build_tree(manage, tmp_path, monkeypatch):
    """No llama.cpp ggml to copy means nothing was invalidated."""
    builder, sd_dir = _staged_swap(manage, tmp_path, monkeypatch)
    (builder.project.src / "llama.cpp").rename(builder.project.src / "llama.cpp.gone")

    builder._sync_ggml_abi()

    assert (sd_dir / "build" / "ggml-metal-device.m.o").exists()


def test_backend_dl_is_off_on_darwin(manage, monkeypatch):
    """Apple emits CMake MODULE libs as MH_BUNDLE, which cannot be linked.

    cyllama's CMakeLists links the ggml backend dylibs directly, so a dynamic
    macOS build has to produce MH_DYLIB -- on every arch and backend, not just
    the x86_64 + Vulkan combination that first hit it.
    """
    monkeypatch.setattr(manage, "PLATFORM", "Darwin")
    assert manage.LlamaCppBuilder._use_backend_dl() is False

    monkeypatch.setattr(manage, "PLATFORM", "Linux")
    assert manage.LlamaCppBuilder._use_backend_dl() is True
