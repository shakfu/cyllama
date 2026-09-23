"""Tests for source-patch application in `scripts/manage.py` (`GgmlBuilder._apply_patch`)."""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANAGE_PY = PROJECT_ROOT / "scripts" / "manage.py"

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


manage = _load_manage()

PATCH = """--- a/src.c
+++ b/src.c
@@ -1,3 +1,3 @@
 int a;
-int b;
+int b = 1;
 int c;
"""


@pytest.fixture
def builder(tmp_path, monkeypatch):
    class FakeBuilder(manage.GgmlBuilder):
        name = "fake.cpp"
        version = "v1"
        repo_url = ""

    monkeypatch.chdir(tmp_path)  # Project() lays out build/ under the cwd
    b = FakeBuilder()
    b.src_dir.mkdir(parents=True)
    return b


def _write(builder, source: str) -> Path:
    (builder.src_dir / "src.c").write_text(source)
    patch = builder.src_dir.parent / "fix.patch"
    patch.write_text(PATCH)
    return patch


def test_applies_matching_patch(builder):
    builder._apply_patch(_write(builder, "int a;\nint b;\nint c;\n"))
    assert (builder.src_dir / "src.c").read_text() == "int a;\nint b = 1;\nint c;\n"


def test_skips_already_applied_patch(builder):
    builder._apply_patch(_write(builder, "int a;\nint b = 1;\nint c;\n"))
    assert (builder.src_dir / "src.c").read_text() == "int a;\nint b = 1;\nint c;\n"


def test_non_matching_patch_fails_the_build(builder, caplog):
    patch = _write(builder, "int a;\nint moved;\nint c;\n")
    with pytest.raises(SystemExit) as exc:
        builder._apply_patch(patch)
    assert exc.value.code == 1
    assert "fix.patch no longer applies to fake.cpp v1" in caplog.text
    assert "patch failed" in caplog.text  # git's own reason is included
    assert (builder.src_dir / "src.c").read_text() == "int a;\nint moved;\nint c;\n"


def _patches_seen(monkeypatch, builder_cls) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(manage.GgmlBuilder, "_apply_patch", lambda self, p: seen.append(p.name))
    builder_cls()._apply_source_patches()
    return seen


@pytest.mark.parametrize("vendored", ["0", "1"])
def test_sd_never_gets_ggml_patches(tmp_path, monkeypatch, vendored):
    # Shared mode compiles llama.cpp's patched tree; vendored mode compiles
    # leejet's fork, which upstream-ggml patches do not match.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SD_USE_VENDORED_GGML", vendored)
    monkeypatch.setattr(manage.GgmlBuilder, "_apply_patch", lambda self, p: seen.append(p.name))
    seen: list[str] = []
    manage.StableDiffusionCppBuilder()._apply_source_patches()
    assert list((Path(manage.__file__).parent / "patches").glob("ggml-*.patch"))  # not vacuous
    assert not any(n.startswith("ggml-") for n in seen)


@pytest.mark.parametrize("cls_name", ["LlamaCppBuilder", "WhisperCppBuilder"])
def test_other_trees_always_get_ggml_patches(tmp_path, monkeypatch, cls_name):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "0")
    seen = _patches_seen(monkeypatch, getattr(manage, cls_name))
    assert any(n.startswith("ggml-") for n in seen)
