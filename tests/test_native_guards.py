"""Inputs that llama.cpp rejects with GGML_ASSERT / GGML_ABORT, which kill the
process. The wrapper must raise first. Each call here aborted before its guard."""

import os

import pytest

import cyllama.llama.llama_cpp as cy

PROMPT = "The capital of France is"


@pytest.fixture(scope="module")
def model(model_path):
    return cy.LlamaModel(model_path, verbose=False)


def _ctx(model):
    p = cy.LlamaContextParams()
    p.n_ctx = 128
    return cy.LlamaContext(model, p)


class TestMemorySeqId:
    @pytest.mark.parametrize(
        "call",
        [
            lambda c: c.memory_seq_pos_max(300),
            lambda c: c.memory_seq_pos_min(-1),
            lambda c: c.memory_seq_keep(1),
            lambda c: c.memory_seq_add(5, 0, -1, 1),
            lambda c: c.memory_seq_div(5, 0, -1, 2),
            lambda c: c.memory_seq_cp(0, 9, 0, -1),
            lambda c: c.memory_seq_rm(-2, 0, -1),
        ],
        ids=["pos_max", "pos_min", "keep", "add", "div", "cp", "rm"],
    )
    def test_out_of_range_raises(self, model, call):
        with pytest.raises(IndexError, match="n_seq_max"):
            call(_ctx(model))

    def test_rm_all_sequences_allowed(self, model):
        ctx = _ctx(model)
        tokens = model.get_vocab().tokenize(PROMPT, add_special=True, parse_special=False)
        ctx.decode(cy.llama_batch_get_one(tokens))
        assert ctx.memory_seq_rm(-1, 0, -1)
        assert ctx.memory_seq_pos_max(0) < 0

    def test_closed_context_raises(self, model):
        ctx = _ctx(model)
        ctx.close()
        with pytest.raises(RuntimeError, match="closed"):
            ctx.memory_seq_pos_max(0)


class TestLoadMode:
    def test_invalid_value_rejected(self):
        p = cy.LlamaModelParams()
        with pytest.raises(ValueError, match="unknown load mode"):
            p.load_mode = 99
        assert p.load_mode_name  # still readable

    def test_valid_values_accepted(self):
        p = cy.LlamaModelParams()
        for mode in (cy.LLAMA_LOAD_MODE_AUTO, cy.LLAMA_LOAD_MODE_NONE, cy.LLAMA_LOAD_MODE_DIRECT_IO):
            p.load_mode = mode
            assert p.load_mode == mode


class TestChainLink:
    """chain_get() returns a link; the llama_sampler_chain_* functions cast any
    sampler to a chain without checking, so using a link as one corrupts memory."""

    def _link(self):
        s = cy.LlamaSampler()
        s.add_top_k(40)
        s.add_greedy()
        return s, s.chain_get(0)

    def test_len_of_link_is_zero(self):
        _, link = self._link()
        assert len(link) == 0

    def test_add_to_link_rejected(self):
        _, link = self._link()
        with pytest.raises(ValueError, match="not a chain"):
            link.add_greedy()

    @pytest.mark.parametrize("op", ["chain_get", "chain_remove"])
    def test_chain_ops_on_link_rejected(self, op):
        _, link = self._link()
        with pytest.raises(ValueError, match="not a chain"):
            getattr(link, op)(0)


def _small_ctx(model, n_batch=16, n_ctx=256):
    p = cy.LlamaContextParams()
    p.n_ctx = n_ctx
    p.n_batch = n_batch
    p.n_ubatch = n_batch
    return cy.LlamaContext(model, p)


LONG_PROMPT = "The quick brown fox jumps over the lazy dog. " * 8  # > 16 tokens


class TestDecodeBatchSize:
    """llama_decode GGML_ASSERTs n_tokens <= n_batch (and <= n_ubatch when non-causal)."""

    def test_oversized_batch_raises(self, model):
        ctx = _small_ctx(model)
        tokens = model.get_vocab().tokenize(LONG_PROMPT, add_special=True, parse_special=False)
        assert len(tokens) > ctx.n_batch
        with pytest.raises(ValueError, match="exceeds the context's n_batch"):
            ctx.decode(cy.llama_batch_get_one(tokens))

    def test_non_causal_checks_n_ubatch(self, model):
        p = cy.LlamaContextParams()
        p.n_ctx = 256
        p.n_batch = 64
        p.n_ubatch = 16
        ctx = cy.LlamaContext(model, p)
        ctx.set_causal_attn(False)
        tokens = model.get_vocab().tokenize(LONG_PROMPT, add_special=True, parse_special=False)[:32]
        with pytest.raises(ValueError, match="n_ubatch"):
            ctx.decode(cy.llama_batch_get_one(tokens))

    def test_none_batch_raises(self, model):
        with pytest.raises(TypeError):
            _small_ctx(model).decode(None)

    def test_llm_reused_context_chunks_by_its_own_n_batch(self, model_path):
        # The context is created with n_batch=16; a later config asks for more.
        import gc

        from cyllama import LLM, GenerationConfig

        llm = LLM(model_path, n_batch=16, n_ctx=256, max_tokens=2, temperature=0.0, verbose=False)
        llm(LONG_PROMPT)
        out = llm(LONG_PROMPT, config=GenerationConfig(n_batch=512, n_ctx=256, max_tokens=2, temperature=0.0))
        assert isinstance(str(out), str)
        llm.close()
        del llm
        gc.collect()

    def test_server_slot_chunks_long_prompt(self, model, model_path):
        from cyllama.llama.server.python import ServerConfig, ServerSlot

        slot = ServerSlot(0, model, ServerConfig(model_path=model_path, n_ctx=256, n_batch=16))
        list(slot.stream(LONG_PROMPT, max_tokens=2))  # aborted the process before chunking
        slot.reset()


class TestSamplerNullLink:
    def test_invalid_grammar_raises(self, model):
        s = cy.LlamaSampler()
        with pytest.raises(ValueError, match="invalid grammar"):
            s.add_grammar(model.get_vocab(), "root ::= (", "root")
        assert len(s) == 0

    def test_missing_root_raises(self, model):
        with pytest.raises(ValueError, match="no rule named 'start'"):
            cy.LlamaSampler().add_grammar(model.get_vocab(), 'root ::= "a"', "start")

    def test_invalid_trigger_regex_raises(self, model):
        with pytest.raises(RuntimeError):
            cy.LlamaSampler().add_grammar_lazy_patterns(
                model.get_vocab(), 'root ::= "a"', "root", trigger_patterns=["(unclosed"]
            )

    def test_grammar_rejecting_accept_raises(self, model):
        vocab = model.get_vocab()
        s = cy.LlamaSampler()
        s.add_grammar(vocab, 'root ::= "yes"', "root")
        tok = vocab.tokenize("no", add_special=False, parse_special=False)[0]
        with pytest.raises(RuntimeError, match="grammar"):
            s.accept(tok)


class TestTokenRange:
    """llama.cpp reads token tables with .at(); std::out_of_range terminates the process."""

    @pytest.mark.parametrize("method", ["get_text", "get_score", "get_attr", "is_control", "token_to_piece"])
    @pytest.mark.parametrize("token", [-1, 10**7])
    def test_out_of_range_raises(self, model, method, token):
        with pytest.raises(IndexError, match="outside the vocabulary"):
            getattr(model.get_vocab(), method)(token)

    def test_detokenize_out_of_range_raises(self, model):
        with pytest.raises(IndexError, match="outside the vocabulary"):
            model.get_vocab().detokenize([1, 10**7])

    def test_in_range_still_works(self, model):
        vocab = model.get_vocab()
        assert vocab.token_to_piece(vocab.token_bos(), special=True)


class TestTtsHelpersExcept:
    def test_fold_bad_padding_raises(self):
        with pytest.raises(RuntimeError):
            cy.fold([1.0], 10, 4, 2, 8)

    def test_convert_less_than_thousand_out_of_range_raises(self):
        with pytest.raises(IndexError):
            cy.convert_less_than_thousand(2000)


class TestLlamaBatchGuards:
    @pytest.mark.parametrize(
        "kwargs",
        [dict(n_tokens=-1, embd=0, n_seq_max=1), dict(n_tokens=4, embd=0, n_seq_max=0)],
        ids=["n_tokens", "n_seq_max"],
    )
    def test_invalid_sizes_raise(self, kwargs):
        with pytest.raises(ValueError, match="must be >="):
            cy.LlamaBatch(**kwargs)

    def test_zero_tokens_allowed(self):
        assert cy.LlamaBatch(n_tokens=0, embd=0, n_seq_max=1).n_tokens == 0

    @pytest.mark.parametrize("embd", [8, -1])
    def test_embd_mode_rejected(self, embd):
        # it had no token array and no way to fill its embedding array
        with pytest.raises(ValueError, match="LlamaBatchExt"):
            cy.LlamaBatch(n_tokens=4, embd=embd, n_seq_max=1)
        with pytest.raises(ValueError, match="LlamaBatchExt"):
            cy.get_pooled_batch(4, embd=embd)

    def test_embd_defaults_to_zero(self):
        assert cy.LlamaBatch(n_tokens=4, n_seq_max=1).embd == 0

    def test_write_after_close_raises(self):
        b = cy.LlamaBatch(n_tokens=4, embd=0, n_seq_max=1)
        b.close()
        b.close()  # idempotent
        with pytest.raises(RuntimeError, match="closed"):
            b.add(1, 0, [0], True)
        b.set_last_logits_to_true()  # no-op on an empty batch

    def test_decode_after_close_raises(self, model):
        b = cy.LlamaBatch(n_tokens=4, embd=0, n_seq_max=1)
        b.add(1, 0, [0], True)
        b.close()
        with pytest.raises(RuntimeError, match="closed"):
            _small_ctx(model).decode(b)

    def test_closed_batch_not_pooled(self):
        cy.reset_batch_pool()
        b = cy.get_pooled_batch(8)
        b.close()
        cy.return_batch_to_pool(b)
        assert cy.get_batch_pool_stats()["total_pooled_batches"] == 0
        cy.reset_batch_pool()


class TestGGUFContextGuards:
    def test_unopened_raises(self):
        with pytest.raises(RuntimeError, match="closed"):
            cy.GGUFContext().version

    def test_closed_raises_and_repr_safe(self):
        ctx = cy.GGUFContext.empty()
        ctx.close()
        assert "closed" in repr(ctx)
        with pytest.raises(RuntimeError, match="closed"):
            ctx.n_kv

    @pytest.mark.parametrize("call", [lambda c: c.get_key(9999), lambda c: c.get_kv_type(-1)], ids=["key", "kv_type"])
    def test_key_out_of_range_raises(self, call):
        ctx = cy.GGUFContext.empty()
        ctx.set_val_u32("a", 1)
        with pytest.raises(IndexError, match="key id"):
            call(ctx)

    @pytest.mark.parametrize("method", ["get_tensor_name", "get_tensor_type", "get_tensor_offset", "get_tensor_size"])
    def test_tensor_out_of_range_raises(self, method):
        with pytest.raises(IndexError, match="tensor id"):
            getattr(cy.GGUFContext.empty(), method)(0)

    def test_valid_access_still_works(self):
        ctx = cy.GGUFContext.empty()
        ctx.set_val_str("k", "v")
        assert ctx.get_key(0) == "k"
        assert ctx.get_value("k") == "v"


class TestSDImageGuards:
    def test_truncated_ppm_raises(self, tmp_path):
        from cyllama.sd import SDImage

        path = tmp_path / "trunc.ppm"
        path.write_bytes(b"P6\n64 64\n255\n" + b"\x00" * 10)
        with pytest.raises(ValueError, match="Truncated PPM"):
            SDImage.load_ppm(str(path))

    def test_complete_ppm_loads(self, tmp_path):
        from cyllama.sd import SDImage

        path = tmp_path / "ok.ppm"
        path.write_bytes(b"P6\n2 2\n255\n" + bytes(range(12)))
        img = SDImage.load_ppm(str(path))
        assert (img.width, img.height) == (2, 2)

    @pytest.mark.parametrize("setter", ["set_ref_images", "set_pm_id_images"])
    def test_non_sdimage_element_raises(self, setter):
        import numpy as np

        from cyllama.sd import SDImage, SDImageGenParams

        params = SDImageGenParams()
        img = SDImage.from_numpy(np.zeros((4, 4, 3), dtype=np.uint8))
        getattr(params, setter)([img])
        with pytest.raises(TypeError, match=r"images\[1\]"):
            getattr(params, setter)([img, np.zeros((4, 4, 3), dtype=np.uint8)])
        getattr(params, setter)([])  # clearing still works


SMALL_EMBED_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "bge-small-en-v1.5-q8_0.gguf")


@pytest.mark.skipif(not os.path.exists(SMALL_EMBED_MODEL), reason="bge-small not downloaded")
def test_embedder_truncates_to_n_batch():
    # bge is non-causal: a text longer than n_ubatch aborted the process.
    import gc

    from cyllama.rag.embedder import Embedder

    emb = Embedder(SMALL_EMBED_MODEL, n_ctx=512, n_batch=32, n_gpu_layers=0)
    assert len(emb.embed(LONG_PROMPT * 4)) == emb.dimension
    emb.close()
    del emb
    gc.collect()


class TestSDListSettersKeepStateOnError:
    """A bad element used to free the old buffer and leave the params pointing at it."""

    def test_slg_layers(self):
        from cyllama.sd import SDSampleParams

        p = SDSampleParams()
        p.slg_layers = [7, 8]
        with pytest.raises((TypeError, ValueError)):
            p.slg_layers = [1, "x"]
        assert p.slg_layers == [7, 8]
        p.slg_layers = []
        assert p.slg_layers == []

    def test_custom_sigmas(self):
        from cyllama.sd import SDSampleParams

        p = SDSampleParams()
        p.custom_sigmas = [1.5, 0.5]
        with pytest.raises((TypeError, ValueError)):
            p.custom_sigmas = [1.0, object()]
        assert p.custom_sigmas == pytest.approx([1.5, 0.5])

    def test_loras(self):
        from cyllama.sd import SDImageGenParams

        p = SDImageGenParams()
        p.set_loras([{"path": "a.safetensors", "multiplier": 0.5}])
        with pytest.raises(KeyError, match="path"):
            p.set_loras([{"path": "b.safetensors"}, {"multiplier": 1.0}])
        str(p)  # reads the params struct; there is no LoRA getter to compare against
        p.set_loras([{"path": "c.safetensors"}])
        p.set_loras([])


class TestSmallBindingFixes:
    def test_detokenize_keeps_leading_whitespace(self, model):
        # it returned result.lstrip()
        vocab = model.get_vocab()
        text = "  indented\nline"
        tokens = vocab.tokenize(text, add_special=False, parse_special=False)
        assert vocab.detokenize(tokens) == text

    def test_abort_callback_none_clears(self, model, capfd):
        # None was installed as the callable; every poll printed a TypeError
        ctx = _small_ctx(model, n_batch=64)
        ctx.set_abort_callback(lambda: False)
        ctx.set_abort_callback(None)
        tokens = model.get_vocab().tokenize("Hello", add_special=True, parse_special=False)
        ctx.decode(cy.llama_batch_get_one(tokens))
        assert "TypeError" not in capfd.readouterr().err

    def test_context_default_params(self):
        # returned LlamaContext(), which raises without a model
        assert isinstance(cy.LlamaContext.default_params(), cy.LlamaContextParams)

    def test_progress_callback_returning_none_continues(self, model_path):
        # None converted to False and aborted the load
        import gc

        seen = []
        params = cy.LlamaModelParams()
        params.n_gpu_layers = 0
        params.progress_callback = lambda p: seen.append(p)
        m = cy.LlamaModel(model_path, params, verbose=False)
        assert seen and m.get_vocab().n_vocab > 0
        m.close()
        del m
        gc.collect()


class TestSDImageFromNumpy:
    def test_float_array_rejected(self):
        import numpy as np

        from cyllama.sd import SDImage

        # [0, 1] floats became an image of 0s and 1s
        with pytest.raises(TypeError, match="integer pixels"):
            SDImage.from_numpy(np.full((4, 4, 3), 0.5))

    def test_bad_shape_rejected_before_allocating(self):
        import numpy as np

        from cyllama.sd import SDImage

        with pytest.raises(ValueError, match="shape"):
            SDImage.from_numpy(np.zeros((2, 4, 4, 3), dtype=np.uint8))

    def test_out_of_range_rejected(self):
        import numpy as np

        from cyllama.sd import SDImage

        with pytest.raises(ValueError, match=r"\[0, 255\]"):
            SDImage.from_numpy(np.full((2, 2), 300, dtype=np.int32))

    def test_uint8_and_small_ints_ok(self):
        import numpy as np

        from cyllama.sd import SDImage

        assert SDImage.from_numpy(np.zeros((2, 2, 3), dtype=np.uint8)).width == 2
        assert SDImage.from_numpy(np.full((2, 2), 7, dtype=np.int64)).channels == 1


def test_model_load_releases_the_gil(model_path):
    # llama_model_load_from_file ran holding the GIL, so other Python threads stopped for the whole load
    import gc
    import threading
    import time

    ticks = []
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ticks.append(time.perf_counter())

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    time.sleep(0.05)
    start = time.perf_counter()
    m = cy.LlamaModel(model_path, verbose=False)
    end = time.perf_counter()
    stop.set()
    t.join()
    during = [x for x in ticks if start < x < end]
    # the longest pause the spinning thread saw inside the load window
    gaps = [b - a for a, b in zip(during, during[1:])] or [end - start]
    assert len(during) > 100 and max(gaps) < 0.5 * (end - start)
    m.close()
    del m
    gc.collect()
