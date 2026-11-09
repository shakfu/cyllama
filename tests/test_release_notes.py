"""Tests for scripts/release_notes.py."""

import importlib.util
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("release_notes", ROOT / "scripts" / "release_notes.py")
assert _spec is not None and _spec.loader is not None
release_notes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release_notes)

PINS = {"llama_cpp": "b1", "whisper_cpp": "v1", "stable_diffusion_cpp": "m", "sqlite_vector": "1.0"}


def test_upstream_versions():
    versions = release_notes.upstream_versions(PINS)
    assert versions[0] == ("llama.cpp", "b1", "https://github.com/ggml-org/llama.cpp/releases/tag/b1")
    assert [tag for _, tag, _ in versions] == ["b1", "v1", "m", "1.0"]


def test_upstream_versions_missing_key():
    pins = dict(PINS)
    del pins["stable_diffusion_cpp"]
    with pytest.raises(KeyError, match="stable_diffusion_cpp"):
        release_notes.upstream_versions(pins)


def test_versions_toml_covers_upstreams():
    pins = tomllib.loads((ROOT / "scripts" / "versions.toml").read_text())
    assert set(pins) == {key for key, _, _ in release_notes.UPSTREAMS}
    assert all(isinstance(v, str) and v for v in pins.values())


def test_main_appends_upstreams(tmp_path):
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## [1.0.0]\n\n- thing\n\n## [0.9.0]\n\n- old\n")
    versions = tmp_path / "versions.toml"
    versions.write_text("".join(f'{k} = "{v}"\n' for k, v in PINS.items()))
    out = tmp_path / "notes.md"
    rc = release_notes.main(["1.0.0", "--changelog", str(changelog), "--versions", str(versions), "-o", str(out)])
    assert rc == 0
    assert out.read_text() == (
        "## Changes since the last Release\n\n- thing\n\n## Upstream versions\n\n"
        "- llama.cpp: [b1](https://github.com/ggml-org/llama.cpp/releases/tag/b1)\n"
        "- whisper.cpp: [v1](https://github.com/ggml-org/whisper.cpp/releases/tag/v1)\n"
        "- stable-diffusion.cpp: [m](https://github.com/leejet/stable-diffusion.cpp/releases/tag/m)\n"
        "- sqlite-vector: [1.0](https://github.com/sqliteai/sqlite-vector/releases/tag/1.0)\n"
    )
