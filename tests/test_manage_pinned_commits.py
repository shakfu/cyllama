"""`scripts/manage.py` refuses a checkout whose HEAD is not the commit versions.toml pins."""

import subprocess

import pytest

from test_manage_env_flags import manage


def _repo(path):
    path.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


@pytest.fixture
def pins(monkeypatch):
    monkeypatch.setitem(manage._VERSIONS, "llama_cpp", "b1")
    monkeypatch.setattr(manage, "PINNED_COMMITS", {})
    return manage.PINNED_COMMITS


def test_matching_commit_passes(tmp_path, pins):
    pins["llama_cpp"] = _repo(tmp_path / "src")
    manage.verify_commit(tmp_path / "src", "llama_cpp", "b1")


def test_moved_tag_raises(tmp_path, pins):
    _repo(tmp_path / "src")
    pins["llama_cpp"] = "0" * 40
    with pytest.raises(RuntimeError, match="pins 0000"):
        manage.verify_commit(tmp_path / "src", "llama_cpp", "b1")


def test_non_pinned_version_not_checked(tmp_path, pins):
    # A --version override has no recorded commit to compare against.
    _repo(tmp_path / "src")
    pins["llama_cpp"] = "0" * 40
    manage.verify_commit(tmp_path / "src", "llama_cpp", "b2")


def test_builder_removes_unverified_tree(tmp_path, pins, monkeypatch):
    _repo(tmp_path / "src")
    pins["llama_cpp"] = "0" * 40
    builder = manage.LlamaCppBuilder(version="b1")
    monkeypatch.setattr(type(builder), "src_dir", property(lambda self: tmp_path / "src"))
    with pytest.raises(RuntimeError):
        builder._verify_commit()
    assert not (tmp_path / "src").exists()


def test_shipped_pins_cover_every_builder():
    for cls in (
        manage.LlamaCppBuilder,
        manage.WhisperCppBuilder,
        manage.StableDiffusionCppBuilder,
        manage.SqliteVectorBuilder,
    ):
        assert manage.pinned_commit(cls.pin_key, cls.version), cls.__name__


def test_clean_refuses_another_directory(tmp_path, monkeypatch):
    # Project() follows the cwd; clean --reset from elsewhere deleted that tree's .venv, build/ and dist/
    import argparse

    monkeypatch.chdir(tmp_path)
    (tmp_path / "build").mkdir()
    with pytest.raises(SystemExit):
        manage.Application().do_clean(argparse.Namespace(verbose=False, reset=True))
    assert (tmp_path / "build").exists()


@pytest.mark.parametrize("member", ["../evil.txt", "dest_evil/x.txt"])
def test_extract_refuses_escaping_members(tmp_path, member):
    # a string-prefix check accepted a sibling such as dest_evil/ for dest/
    import io
    import tarfile

    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tar:
        data = b"x"
        info = tarfile.TarInfo(("../" if member.startswith("dest_evil") else "") + member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(ValueError, match="path traversal"):
        manage.ShellCmd().extract(archive, dest)


def test_extract_refuses_escaping_symlink(tmp_path):
    import tarfile

    archive = tmp_path / "a.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(tarfile.FilterError):
        manage.ShellCmd().extract(archive, dest)


@pytest.mark.parametrize(
    "current,kw,expected",
    [
        ("1.0.0a0", {"pre": True}, "1.0.0a1"),
        ("1.0.0rc2", {"pre": True}, "1.0.0rc3"),
        ("1.0.0a1", {"release": True}, "1.0.0"),
        ("1.0.0", {}, "1.0.1"),
        ("1.2.3", {"minor": True}, "1.3.0"),
        ("1.2.3", {"major": True}, "2.0.0"),
    ],
)
def test_next_version(current, kw, expected):
    # bump parsed only X.Y.Z, so it could not bump the current 1.0.0a0
    assert manage.next_version(current, **kw) == expected


@pytest.mark.parametrize(
    "current,kw",
    [
        ("1.0.0a0", {}),
        ("1.0.0", {"pre": True}),
        ("1.0.0", {"release": True}),
        ("1.0", {}),
        ("1.0.0", {"major": True, "pre": True}),
    ],
)
def test_next_version_refuses(current, kw):
    with pytest.raises(ValueError):
        manage.next_version(current, **kw)


def test_versions_agree():
    # Makefile (0.1.20) and manage.py --version (0.0.4) disagreed with pyproject.toml
    import re
    from pathlib import Path

    root = Path(manage.__file__).resolve().parent.parent
    init = (root / "src" / "cyllama" / "__init__.py").read_text()
    assert manage.project_version() == re.search(r'__version__ = "([^"]+)"', init).group(1)
    assert manage.Application().version == manage.project_version()
    assert "VERSION :=" not in (root / "Makefile").read_text()


def test_one_macos_deployment_target():
    from pathlib import Path

    root = Path(manage.__file__).resolve().parent.parent
    target = manage.MACOSX_DEPLOYMENT_TARGET_DEFAULT
    assert f"MACOSX_DEPLOYMENT_TARGET := {target}" in (root / "Makefile").read_text()
    assert f'MACOSX_DEPLOYMENT_TARGET = "{target}"' in (root / "pyproject.toml").read_text()
