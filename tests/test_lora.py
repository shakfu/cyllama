"""LoRA adapter surface: loading, lifetime, and applying to a context.

These tests build their own GGUF adapters. A real LoRA is model-specific and
none ships with the repo, so the whole surface was previously untestable and
therefore untested. The writer below emits the minimum llama.cpp's loader
accepts -- four metadata keys and one lora_a/lora_b tensor pair against a
single base-model tensor -- which is enough to exercise every wrapper path
including a real decode. Weights are zero, so the adapter is a no-op
numerically; these tests are about the wrapper, not about output quality.
"""

import gc
import os
import struct

import pytest

import cyllama.llama.llama_cpp as cy

# GGUF metadata value types and the one ggml tensor type used here.
_T_UINT32, _T_FLOAT32, _T_STRING, _T_ARRAY = 4, 6, 8, 9
_GGML_F32 = 0
_ALIGN = 32

# blk.0.attn_q.weight is [n_embd, n_embd] for Llama-3.2-1B.
_N_EMBD = 2048
_BASE_TENSOR = "blk.0.attn_q.weight"


def _gguf_str(raw: bytes) -> bytes:
    return struct.pack("<Q", len(raw)) + raw


def _kv_str(key: str, value: str) -> bytes:
    return _gguf_str(key.encode()) + struct.pack("<I", _T_STRING) + _gguf_str(value.encode())


def _kv_f32(key: str, value: float) -> bytes:
    return _gguf_str(key.encode()) + struct.pack("<I", _T_FLOAT32) + struct.pack("<f", value)


def _kv_u32_array(key: str, values) -> bytes:
    return (
        _gguf_str(key.encode())
        + struct.pack("<I", _T_ARRAY)
        + struct.pack("<I", _T_UINT32)
        + struct.pack("<Q", len(values))
        + b"".join(struct.pack("<I", v) for v in values)
    )


def write_lora_gguf(path, *, alpha=8.0, rank=1, arch="llama", invocation_tokens=None):
    """Write a minimal GGUF LoRA adapter and return its path.

    Shapes follow llama.cpp's validation for a non-embedding tensor:
    ``a.ne == (n_in, rank)``, ``b.ne == (rank, n_out)``.
    """
    kv = [
        _kv_str("general.architecture", arch),
        _kv_str("general.type", "adapter"),
        _kv_str("adapter.type", "lora"),
        _kv_f32("adapter.lora.alpha", alpha),
    ]
    if invocation_tokens:
        kv.append(_kv_u32_array("adapter.alora.invocation_tokens", invocation_tokens))

    tensors = [
        (f"{_BASE_TENSOR}.lora_a", (_N_EMBD, rank)),
        (f"{_BASE_TENSOR}.lora_b", (rank, _N_EMBD)),
    ]

    def nbytes(ne):
        total = 4
        for dim in ne:
            total *= dim
        return total

    infos, offset = b"", 0
    for name, ne in tensors:
        infos += (
            _gguf_str(name.encode())
            + struct.pack("<I", len(ne))
            + b"".join(struct.pack("<Q", d) for d in ne)
            + struct.pack("<I", _GGML_F32)
            + struct.pack("<Q", offset)
        )
        offset += -(-nbytes(ne) // _ALIGN) * _ALIGN

    header = (
        b"GGUF"
        + struct.pack("<I", 3)
        + struct.pack("<Q", len(tensors))
        + struct.pack("<Q", len(kv))
        + b"".join(kv)
        + infos
    )

    with open(path, "wb") as f:
        f.write(header + b"\0" * (-len(header) % _ALIGN))
        for _, ne in tensors:
            size = nbytes(ne)
            f.write(b"\0" * size)
            f.write(b"\0" * (-size % _ALIGN))
    return str(path)


@pytest.fixture(scope="module")
def lora_path(tmp_path_factory):
    return write_lora_gguf(tmp_path_factory.mktemp("lora") / "adapter.gguf")


@pytest.fixture(scope="module")
def alora_path(tmp_path_factory):
    return write_lora_gguf(
        tmp_path_factory.mktemp("alora") / "adapter.gguf",
        invocation_tokens=[128000, 42, 7],
    )


@pytest.fixture(scope="module")
def model(model_path):
    m = cy.LlamaModel(model_path)
    yield m
    del m
    gc.collect()


class TestAdapterLoading:
    def test_load_and_read_metadata(self, model, lora_path):
        adapter = model.lora_adapter_init(lora_path)

        assert adapter.meta_count() == 4
        assert adapter.meta_val_str("adapter.type") == "lora"
        assert adapter.meta_val_str("general.type") == "adapter"
        keys = {adapter.meta_key_by_index(i) for i in range(adapter.meta_count())}
        assert "adapter.lora.alpha" in keys

    def test_missing_file_raises(self, model, tmp_path):
        with pytest.raises(FileNotFoundError):
            model.lora_adapter_init(str(tmp_path / "absent.gguf"))

    def test_wrong_arch_raises(self, model, tmp_path):
        """The loader rejects an adapter built for another architecture."""
        path = write_lora_gguf(tmp_path / "gpt2.gguf", arch="gpt2")
        with pytest.raises(ValueError):
            model.lora_adapter_init(path)

    def test_close_is_noop_for_model_owned_adapter(self, model, lora_path):
        """Adapters from lora_adapter_init are owned by the model, so close()
        must not free them -- the model's destructor does that."""
        adapter = model.lora_adapter_init(lora_path)

        adapter.close()
        adapter.close()
        # Still usable: close() did not free a pointer it does not own.
        assert adapter.meta_count() == 4


class TestAdapterFromFileobj:
    def test_embedded_at_offset(self, model, lora_path, tmp_path):
        bundle = tmp_path / "bundle.bin"
        with open(lora_path, "rb") as src:
            bundle.write_bytes(b"\xab" * 100 + src.read() + b"\xcd" * 16)

        with open(bundle, "rb") as f:
            adapter = model.lora_adapter_init_from_fileobj(f, offset=100)
            assert f.tell() == 0

        assert adapter.meta_val_str("adapter.type") == "lora"
        assert adapter.model is model

    def test_raw_fd_at_current_position(self, model, lora_path):
        fd = os.open(lora_path, os.O_RDONLY)
        try:
            adapter = model.lora_adapter_init_from_fileobj(fd)
        finally:
            os.close(fd)
        assert adapter.meta_count() == 4

    def test_wrong_arch_raises(self, model, tmp_path):
        path = write_lora_gguf(tmp_path / "gpt2.gguf", arch="gpt2")
        with open(path, "rb") as f, pytest.raises(ValueError, match="Failed to load LoRA"):
            model.lora_adapter_init_from_fileobj(f)

    def test_bad_header_raises(self, model, tmp_path):
        path = tmp_path / "junk.bin"
        path.write_bytes(b"\0" * 64)
        with open(path, "rb") as f, pytest.raises(ValueError, match="valid GGUF"):
            model.lora_adapter_init_from_fileobj(f)


class TestAdapterLifetime:
    """llama_model's destructor deletes every adapter registered to it, so an
    adapter that does not retain its model dangles as soon as the model is
    collected."""

    def test_adapter_retains_its_model(self, model, lora_path):
        adapter = model.lora_adapter_init(lora_path)
        assert adapter.model is model

    def test_survives_temporary_model(self, model_path, lora_path):
        """The natural one-liner: the model is a temporary with no other
        reference. Without the retained parent the adapter would dereference
        freed memory on the next line."""
        adapter = cy.LlamaModel(model_path).lora_adapter_init(lora_path)
        gc.collect()

        assert adapter.meta_val_str("adapter.type") == "lora"
        assert adapter.model is not None

        del adapter
        gc.collect()


class TestAloraInvocationTokens:
    def test_plain_lora_has_no_invocation_sequence(self, model, lora_path):
        adapter = model.lora_adapter_init(lora_path)

        assert adapter.n_alora_invocation_tokens == 0
        assert adapter.alora_invocation_tokens == []

    def test_alora_exposes_its_sequence(self, model, alora_path):
        adapter = model.lora_adapter_init(alora_path)

        assert adapter.n_alora_invocation_tokens == 3
        assert adapter.alora_invocation_tokens == [128000, 42, 7]


class TestSetAdaptersLora:
    """`llama_set_adapters_lora` has set-semantics, not add-semantics: each
    call replaces the whole set."""

    @pytest.fixture
    def ctx(self, model):
        params = cy.LlamaContextParams()
        params.n_ctx = 128
        c = cy.LlamaContext(model, params)
        yield c
        c.close()
        del c
        gc.collect()

    def test_fresh_context_has_none(self, ctx):
        assert ctx.lora_adapters == []

    def test_set_then_clear(self, ctx, model, lora_path):
        adapter = model.lora_adapter_init(lora_path)

        ctx.set_adapters_lora([(adapter, 1.0)])
        assert ctx.lora_adapters == [adapter]

        ctx.set_adapters_lora()
        assert ctx.lora_adapters == []

    def test_set_replaces_rather_than_appends(self, ctx, model, lora_path, alora_path):
        first = model.lora_adapter_init(lora_path)
        second = model.lora_adapter_init(alora_path)

        ctx.set_adapters_lora([(first, 1.0)])
        ctx.set_adapters_lora([(second, 1.0)])

        assert ctx.lora_adapters == [second]

    def test_accepts_a_mapping(self, ctx, model, lora_path, alora_path):
        first = model.lora_adapter_init(lora_path)
        second = model.lora_adapter_init(alora_path)

        ctx.set_adapters_lora({first: 0.5, second: 0.25})

        assert set(ctx.lora_adapters) == {first, second}

    def test_decode_with_adapters_applied(self, ctx, model, lora_path):
        adapter = model.lora_adapter_init(lora_path)
        ctx.set_adapters_lora([(adapter, 1.0)])

        tokens = model.get_vocab().tokenize("Hello", add_special=True, parse_special=False)
        batch = cy.LlamaBatch(n_tokens=len(tokens), embd=0, n_seq_max=1)
        batch.set_batch(tokens, 0, False)

        assert ctx.decode(batch) == 0

    def test_duplicate_adapter_rejected(self, ctx, model, lora_path):
        """llama.cpp keys adapters by pointer, so a repeat would silently
        drop one of the two scales."""
        adapter = model.lora_adapter_init(lora_path)

        with pytest.raises(ValueError, match="more than once"):
            ctx.set_adapters_lora([(adapter, 1.0), (adapter, 0.5)])

    def test_retains_previous_set_when_rejected(self, ctx, model, lora_path):
        adapter = model.lora_adapter_init(lora_path)
        ctx.set_adapters_lora([(adapter, 1.0)])

        with pytest.raises(ValueError):
            ctx.set_adapters_lora([(adapter, 1.0), (adapter, 0.5)])

        assert ctx.lora_adapters == [adapter]

    @pytest.mark.parametrize(
        "bad",
        ["not-a-pair", [("path", 1.0)], [(1, 2, 3)], [None]],
        ids=["string", "wrong-type", "triple", "none"],
    )
    def test_rejects_malformed_entries(self, ctx, bad):
        with pytest.raises(TypeError):
            ctx.set_adapters_lora(bad)

    def test_context_retains_adapters(self, ctx, model, lora_path):
        """The C side keeps raw pointers only. The context must hold the
        wrappers so an adapter borrowed from a model the caller has dropped
        stays alive."""
        ctx.set_adapters_lora([(model.lora_adapter_init(lora_path), 1.0)])
        gc.collect()

        assert len(ctx.lora_adapters) == 1
        assert ctx.lora_adapters[0].meta_count() == 4


class TestCliLoraFlags:
    """`--lora` / `--lora-scaled` parsed and resolved without loading a model.

    Both flags existed in the CLI but were never read: nothing between
    argparse and context creation touched them, so passing either silently
    did nothing.
    """

    def _parse(self, argv):
        import sys
        from unittest.mock import patch

        from cyllama.llama.cli import LlamaCLI

        with patch.object(sys, "argv", ["llama-cli", "-m", "model.gguf", *argv]):
            return LlamaCLI._parse_args(object.__new__(LlamaCLI))

    def _resolve(self, args, loaded):
        """Run the resolver against a model stub that records load order."""
        from cyllama.llama.cli import LlamaCLI

        cli = object.__new__(LlamaCLI)

        class _Model:
            def lora_adapter_init(self, path):
                loaded.append(path)
                return path

        cli.model = _Model()
        return LlamaCLI._load_lora_adapters(cli, args)

    def test_no_flags_resolves_to_nothing(self):
        loaded = []
        assert self._resolve(self._parse([]), loaded) == []
        assert loaded == []

    def test_lora_defaults_to_scale_one(self):
        loaded = []
        pairs = self._resolve(self._parse(["--lora", "a.gguf"]), loaded)

        assert pairs == [("a.gguf", 1.0)]
        assert loaded == ["a.gguf"]

    def test_flags_are_repeatable(self):
        args = self._parse(["--lora", "a.gguf", "--lora", "b.gguf", "--lora-scaled", "c.gguf", "0.5"])
        pairs = self._resolve(args, [])

        assert pairs == [("a.gguf", 1.0), ("b.gguf", 1.0), ("c.gguf", 0.5)]

    def test_bad_scale_fails_before_loading_anything(self):
        loaded = []
        args = self._parse(["--lora", "a.gguf", "--lora-scaled", "b.gguf", "loud"])

        with pytest.raises(SystemExit, match="must be a number"):
            self._resolve(args, loaded)
        assert loaded == []

    def test_lora_base_flag_is_gone(self):
        """--lora-base named the pre-adapter apply-to-weights path, which
        llama.cpp removed; accepting it would promise something unimplementable."""
        with pytest.raises(SystemExit):
            self._parse(["--lora-base", "base.gguf"])
