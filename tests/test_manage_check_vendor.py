"""`check_vendor` compares ggml headers of whisper.cpp and SD against llama.cpp's tree."""

from types import SimpleNamespace

import pytest

from test_manage_env_flags import manage


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.mark.parametrize("name", ["whisper.cpp", "stable-diffusion.cpp"])
def test_ggml_headers_checked_against_llama(tmp_path, name):
    # Both build against llama.cpp's ggml, so their own bundled ggml.h is irrelevant.
    committed = tmp_path / "committed"
    _write(committed / "ggml.h", b"llama ggml")
    _write(committed / "own.h", b"own header")
    clones = {"llama.cpp": tmp_path / "llama", name: tmp_path / "own"}
    _write(clones["llama.cpp"] / "ggml/include/ggml.h", b"llama ggml")
    _write(clones[name] / "ggml/include/ggml.h", b"bundled ggml")
    _write(clones[name] / "include/own.h", b"own header")

    builder = SimpleNamespace(name=name, include=committed)
    app = manage.Application.__new__(manage.Application)
    assert app._check_vendor_by_name(builder, clones) == []

    _write(committed / "ggml.h", b"bundled ggml")
    assert app._check_vendor_by_name(builder, clones) == ["ggml.h: differs from upstream"]
