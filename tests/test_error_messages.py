"""Error message audit tests.

These tests pin down the user-facing exceptions raised when callers pass
bad inputs to model loaders (LLM, LlamaModel, LlamaContext, SDContext,
WhisperContext). The goal is to ensure these failure paths produce clear,
typed Python exceptions instead of segfaults or opaque NULL-pointer errors.

The tests intentionally avoid loading any real model — they only exercise
the validation layer that runs before the underlying C++ libraries are
called. They are therefore safe to run on machines without model files.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_file(tmp_path: Path) -> str:
    p = tmp_path / "empty.gguf"
    p.write_bytes(b"")
    return str(p)


@pytest.fixture
def garbage_file(tmp_path: Path) -> str:
    """A non-empty file that is not a valid GGUF (wrong magic)."""
    p = tmp_path / "garbage.gguf"
    p.write_bytes(b"NOTGGUF" + b"\x00" * 1024)
    return str(p)


@pytest.fixture
def truncated_gguf(tmp_path: Path) -> str:
    """File with the GGUF magic and a full 24-byte header but version=0
    and zero tensors — corrupt enough to be rejected by our validator's
    version check (which is the branch we want to exercise here)."""
    p = tmp_path / "truncated.gguf"
    # 4 bytes magic + 4 bytes version (=0) + 8 bytes tensor_count + 8 bytes kv_count + padding
    p.write_bytes(b"GGUF" + b"\x00" * 20 + b"\x00" * 64)
    return str(p)


@pytest.fixture
def short_gguf(tmp_path: Path) -> str:
    """File with the GGUF magic but fewer than 24 header bytes — exercises
    the 'too small to contain a valid header' branch."""
    p = tmp_path / "short.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 8)
    return str(p)


@pytest.fixture
def unreadable_file(tmp_path: Path) -> str:
    p = tmp_path / "noperm.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 64)
    os.chmod(p, 0)
    yield str(p)
    # restore so tmp_path cleanup can proceed
    os.chmod(p, stat.S_IRUSR | stat.S_IWUSR)


# ---------------------------------------------------------------------------
# _validation helper (the shared building block)
# ---------------------------------------------------------------------------


class TestValidationHelper:
    def test_missing_path(self, tmp_path: Path):
        from cyllama.utils.validation import validate_model_file

        with pytest.raises(FileNotFoundError, match="not found"):
            validate_model_file(str(tmp_path / "does-not-exist.gguf"))

    def test_directory_path(self, tmp_path: Path):
        from cyllama.utils.validation import validate_model_file

        with pytest.raises(IsADirectoryError, match="directory"):
            validate_model_file(str(tmp_path))

    def test_empty_file(self, empty_file: str):
        from cyllama.utils.validation import validate_model_file

        with pytest.raises(ValueError, match="empty"):
            validate_model_file(empty_file)

    def test_wrong_magic(self, garbage_file: str):
        from cyllama.utils.validation import GGUF_MAGIC, validate_model_file

        with pytest.raises(ValueError, match="does not look like a valid"):
            validate_model_file(garbage_file, kind="GGUF model", expected_magic=GGUF_MAGIC)

    def test_unreadable(self, unreadable_file: str):
        from cyllama.utils.validation import validate_model_file

        # On many CI runners the test process is root and ignores chmod 0;
        # in that case the validator can't trip and the assertion is moot.
        if os.access(unreadable_file, os.R_OK):
            pytest.skip("file is still readable (likely running as root)")

        with pytest.raises(PermissionError, match="not readable"):
            validate_model_file(unreadable_file)

    def test_non_string_path(self):
        from cyllama.utils.validation import validate_model_file

        with pytest.raises(TypeError, match="must be a string"):
            validate_model_file(123)  # type: ignore[arg-type]

    def test_empty_string_path(self):
        from cyllama.utils.validation import validate_model_file

        with pytest.raises(ValueError, match="non-empty string"):
            validate_model_file("")

    def test_magic_check_skipped_when_none(self, garbage_file: str):
        from cyllama.utils.validation import validate_model_file

        # With expected_magic=None, a garbage file must validate cleanly
        # (no magic check performed). validate_model_file returns None on
        # success, so the call completing with a None return is the signal.
        assert validate_model_file(garbage_file, expected_magic=None) is None


# ---------------------------------------------------------------------------
# LLM (high-level wrapper) — bad inputs
# ---------------------------------------------------------------------------


class TestLLMErrors:
    def test_missing_model_path(self, tmp_path: Path):
        from cyllama import LLM

        with pytest.raises(FileNotFoundError, match="not found"):
            LLM(str(tmp_path / "nope.gguf"))

    def test_directory_as_model(self, tmp_path: Path):
        from cyllama import LLM

        with pytest.raises(IsADirectoryError, match="directory"):
            LLM(str(tmp_path))

    def test_empty_file(self, empty_file: str):
        from cyllama import LLM

        with pytest.raises(ValueError, match="empty"):
            LLM(empty_file)

    def test_garbage_file(self, garbage_file: str):
        from cyllama import LLM

        with pytest.raises(ValueError, match="does not look like a valid"):
            LLM(garbage_file)


# ---------------------------------------------------------------------------
# LlamaModel (low-level wrapper) — bad inputs
# ---------------------------------------------------------------------------


class TestLlamaModelErrors:
    def test_missing_model_path(self, tmp_path: Path):
        import cyllama.llama.llama_cpp as cy

        with pytest.raises(FileNotFoundError, match="not found"):
            cy.LlamaModel(str(tmp_path / "nope.gguf"))

    def test_directory_as_model(self, tmp_path: Path):
        import cyllama.llama.llama_cpp as cy

        with pytest.raises(IsADirectoryError, match="directory"):
            cy.LlamaModel(str(tmp_path))

    def test_empty_file(self, empty_file: str):
        import cyllama.llama.llama_cpp as cy

        with pytest.raises(ValueError, match="empty"):
            cy.LlamaModel(empty_file)

    def test_garbage_file(self, garbage_file: str):
        import cyllama.llama.llama_cpp as cy

        with pytest.raises(ValueError, match="does not look like a valid"):
            cy.LlamaModel(garbage_file)

    def test_truncated_gguf(self, truncated_gguf: str):
        """A file with valid GGUF magic and a full header but bogus
        version must be rejected by *our* validator's version check, so
        it never reaches llama.cpp's parser."""
        import cyllama.llama.llama_cpp as cy

        with pytest.raises(ValueError, match="unsupported GGUF version"):
            cy.LlamaModel(truncated_gguf)

    def test_short_gguf(self, short_gguf: str):
        """A file with the GGUF magic but fewer than 24 header bytes must
        be rejected by *our* validator's size check, so it never reaches
        llama.cpp's parser."""
        import cyllama.llama.llama_cpp as cy

        with pytest.raises(ValueError, match="too small to contain a valid GGUF header"):
            cy.LlamaModel(short_gguf)


# ---------------------------------------------------------------------------
# LlamaContext — bad inputs
# ---------------------------------------------------------------------------


class TestLlamaContextErrors:
    def test_negative_n_ctx(self):
        """Assigning a negative n_ctx should raise a clear ValueError that
        tells the user about the 0 = "use model default" sentinel — not
        Cython's opaque "can't convert negative value to uint32_t"."""
        import cyllama.llama.llama_cpp as cy

        params = cy.LlamaContextParams()
        with pytest.raises(ValueError, match="n_ctx must be >= 0"):
            params.n_ctx = -1

    def test_n_ctx_uint32_overflow(self):
        """Values that exceed uint32_t should still be caught cleanly
        (Cython's OverflowError) — not silently wrap around."""
        import cyllama.llama.llama_cpp as cy

        params = cy.LlamaContextParams()
        with pytest.raises(OverflowError):
            params.n_ctx = 2**32  # one past uint32 max

    def test_n_ctx_zero_is_valid(self):
        """n_ctx=0 must remain a valid sentinel meaning 'use model default'."""
        import cyllama.llama.llama_cpp as cy

        params = cy.LlamaContextParams()
        params.n_ctx = 0
        assert params.n_ctx == 0

    def test_oom_n_ctx(self, model_path: str):
        """Requesting an absurdly large context must be refused by our
        KV-cache memory pre-check *before* llama.cpp's allocator gets a
        chance to segfault. The check estimates KV memory and rejects
        anything past a 100 TiB sanity cap."""
        import cyllama.llama.llama_cpp as cy

        cy.llama_backend_init()
        try:
            model = cy.LlamaModel(model_path)
            params = cy.LlamaContextParams()
            # 2**31 - 1 tokens × 16 layers × 2048 n_embd × 4 bytes ≈ 280 TiB,
            # well past the 100 TiB cap.
            params.n_ctx = 2_147_483_647
            with pytest.raises(RuntimeError, match="exceeds the .* TiB sanity cap"):
                cy.LlamaContext(model, params)
        finally:
            cy.llama_backend_free()


# ---------------------------------------------------------------------------
# WhisperContext — bad inputs
# ---------------------------------------------------------------------------


class TestWhisperContextErrors:
    def test_missing_model_path(self, tmp_path: Path):
        wh = pytest.importorskip("cyllama.whisper.whisper_cpp")

        with pytest.raises(FileNotFoundError, match="not found"):
            wh.WhisperContext(str(tmp_path / "nope.bin"))

    def test_directory_as_model(self, tmp_path: Path):
        wh = pytest.importorskip("cyllama.whisper.whisper_cpp")

        with pytest.raises(IsADirectoryError, match="directory"):
            wh.WhisperContext(str(tmp_path))

    def test_empty_file(self, empty_file: str):
        wh = pytest.importorskip("cyllama.whisper.whisper_cpp")

        with pytest.raises(ValueError, match="empty"):
            wh.WhisperContext(empty_file)

    def test_garbage_file(self, garbage_file: str):
        """A non-empty file with the wrong magic must be rejected by our
        validator (which checks for one of the known whisper magics:
        GGUF or legacy ggml) so it never reaches whisper.cpp's parser."""
        wh = pytest.importorskip("cyllama.whisper.whisper_cpp")

        with pytest.raises(ValueError, match="does not look like a valid"):
            wh.WhisperContext(garbage_file)


# ---------------------------------------------------------------------------
# SDContext — bad inputs
# ---------------------------------------------------------------------------


class TestSDContextErrors:
    def test_missing_model_path(self, tmp_path: Path):
        sd = pytest.importorskip("cyllama.sd")

        params = sd.SDContextParams(model_path=str(tmp_path / "nope.gguf"))
        with pytest.raises(FileNotFoundError, match="not found"):
            sd.SDContext(params)

    def test_directory_as_model(self, tmp_path: Path):
        sd = pytest.importorskip("cyllama.sd")

        params = sd.SDContextParams(model_path=str(tmp_path))
        with pytest.raises(IsADirectoryError, match="directory"):
            sd.SDContext(params)

    def test_empty_model_file(self, empty_file: str):
        sd = pytest.importorskip("cyllama.sd")

        params = sd.SDContextParams(model_path=empty_file)
        with pytest.raises(ValueError, match="empty"):
            sd.SDContext(params)

    def test_missing_vae_path(self, tmp_path: Path, garbage_file: str):
        """Sub-model paths should also be validated up front."""
        sd = pytest.importorskip("cyllama.sd")

        params = sd.SDContextParams(
            model_path=garbage_file,
            vae_path=str(tmp_path / "missing-vae.safetensors"),
        )
        with pytest.raises(FileNotFoundError, match="VAE"):
            sd.SDContext(params)
