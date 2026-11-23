"""Wrappers that borrow a native pointer keep its owner alive."""

import gc
import sys

import pytest

import cyllama.llama.llama_cpp as cy

GRAMMAR = 'root ::= "yes" | "no"'


def test_vocab_keeps_model_alive(model_path):
    model = cy.LlamaModel(model_path, verbose=False)
    before = sys.getrefcount(model)
    vocab = model.get_vocab()
    assert sys.getrefcount(model) == before + 1
    n_vocab = vocab.n_vocab
    del model
    gc.collect()
    # before the fix the model was freed here and these read freed memory
    assert vocab.n_vocab == n_vocab
    assert vocab.tokenize("hello", add_special=False, parse_special=False)
    del vocab
    gc.collect()


def test_vocab_from_temporary_model(model_path):
    vocab = cy.LlamaModel(model_path, verbose=False).get_vocab()
    gc.collect()
    assert vocab.n_vocab > 0
    del vocab
    gc.collect()


def test_sampler_links_keep_vocab_alive(model_path):
    model = cy.LlamaModel(model_path, verbose=False)
    vocab = model.get_vocab()
    before = sys.getrefcount(vocab)
    sampler = cy.LlamaSampler()
    sampler.add_grammar(vocab, GRAMMAR, "root")
    sampler.add_dry(vocab, 0.8, 1.75, 2, 64, ["\n"])
    sampler.add_infill(vocab)
    assert sys.getrefcount(vocab) == before + 3
    # clones and detached links hold the vocab too
    clone = sampler.clone()
    link = sampler.chain_remove(0)
    assert sys.getrefcount(vocab) == before + 9
    sampler.close()
    clone.close()
    link.close()
    del sampler, clone, link
    gc.collect()
    assert sys.getrefcount(vocab) == before
    del vocab, model
    gc.collect()


def test_grammar_sampler_outlives_its_model_reference(model_path):
    sampler = cy.LlamaSampler()
    sampler.add_grammar(cy.LlamaModel(model_path, verbose=False).get_vocab(), GRAMMAR, "root")
    sampler.add_greedy()
    gc.collect()
    # the grammar link reads the vocab on sample; build a context from a separate model
    model = cy.LlamaModel(model_path, verbose=False)
    params = cy.LlamaContextParams()
    params.n_ctx = 128
    ctx = cy.LlamaContext(model, params)
    vocab = model.get_vocab()
    ctx.decode(cy.llama_batch_get_one(vocab.tokenize("Answer yes or no:", add_special=True, parse_special=False)))
    token = sampler.sample(ctx, -1)
    assert vocab.token_to_piece(token, 0, False) in ("yes", "no", " yes", " no", "y", "n")
    sampler.close()
    ctx.close()
    model.close()
    del sampler, ctx, model, vocab
    gc.collect()


# An explicit close() frees the native object even while Python references remain.
# Every wrapper then raises instead of reading freed memory or dereferencing NULL.


def _stack(model_path):
    model = cy.LlamaModel(model_path, verbose=False)
    params = cy.LlamaContextParams()
    params.n_ctx = 128
    return model, cy.LlamaContext(model, params)


def test_closed_model_raises_everywhere(model_path):
    model, ctx = _stack(model_path)
    vocab = model.get_vocab()
    model.close()
    for call in (lambda: model.n_embd, model.metadata, model.get_vocab):
        with pytest.raises(RuntimeError, match="LlamaModel is closed"):
            call()
    for call in (lambda: vocab.n_vocab, lambda: vocab.tokenize("hi", add_special=False, parse_special=False)):
        with pytest.raises(RuntimeError, match="LlamaModel is closed"):
            call()
    for call in (
        lambda: ctx.n_ctx,
        lambda: ctx.decode(cy.llama_batch_get_one([1])),
        ctx.get_logits,
        ctx.kv_cache_clear,
    ):
        with pytest.raises(RuntimeError, match="LlamaModel is closed"):
            call()
    ctx.close()
    del ctx, vocab, model
    gc.collect()


def test_closed_context_raises(model_path):
    model, ctx = _stack(model_path)
    ctx.close()
    for call in (
        lambda: ctx.n_ctx,
        lambda: ctx.decode(cy.llama_batch_get_one([1])),
        ctx.get_logits,
        ctx.get_state_size,
        lambda: ctx.memory_seq_rm(0, 0, -1),
        lambda: ctx.sampled_token_ith(0),
    ):
        with pytest.raises(RuntimeError, match="LlamaContext is closed"):
            call()
    model.close()
    del ctx, model
    gc.collect()


def test_closed_lora_parent_raises(model_path, tmp_path):
    from test_lora import write_lora_gguf

    model = cy.LlamaModel(model_path, verbose=False)
    adapter = model.lora_adapter_init(str(write_lora_gguf(tmp_path / "adapter.gguf")))
    assert adapter.meta_count() > 0
    model.close()
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        adapter.meta_count()
    del adapter, model
    gc.collect()


def test_closed_sampler_and_chain_view(model_path):
    sampler = cy.LlamaSampler()
    sampler.add_top_k(40)
    sampler.add_greedy()
    view = sampler.chain_get(0)
    before = sys.getrefcount(sampler)
    second = sampler.chain_get(1)
    # a view keeps its chain alive
    assert sys.getrefcount(sampler) == before + 1
    sampler.close()
    with pytest.raises(RuntimeError, match="LlamaSampler is closed"):
        sampler.reset()
    with pytest.raises(RuntimeError, match="chain this link belongs to is closed"):
        view.name()
    del view, second, sampler
    gc.collect()


@pytest.mark.filterwarnings("ignore:Speculative is deprecated:DeprecationWarning")
def test_speculative_with_closed_context(model_path):
    from cyllama.llama.llama_cpp import Speculative, SpeculativeParams

    model, tgt = _stack(model_path)
    params = cy.LlamaContextParams()
    params.n_ctx = 128
    dft = cy.LlamaContext(model, params)
    spec = Speculative(SpeculativeParams(), tgt, dft)
    dft.close()
    with pytest.raises(RuntimeError, match="LlamaContext is closed"):
        spec.draft(SpeculativeParams(), [1, 2], 3)
    with pytest.raises(RuntimeError, match="LlamaContext is closed"):
        Speculative(SpeculativeParams(), tgt, dft)
    tgt.close()
    model.close()
    del spec, tgt, dft, model
    gc.collect()


# llama_free reads the model, so a model's contexts must be freed before it.


def test_model_close_closes_its_contexts(model_path):
    model, ctx = _stack(model_path)
    sampler = cy.LlamaSampler()
    sampler.add_greedy()
    model.close()
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        sampler.sample(ctx, -1)
    ctx.close()  # already freed with the model; idempotent
    del ctx, model, sampler
    gc.collect()


def test_context_in_reference_cycle_is_freed_before_model(model_path):
    model, ctx = _stack(model_path)
    holder = [ctx]
    ctx.set_abort_callback(lambda: holder is None)  # ctx -> callback -> holder -> ctx
    del model, ctx, holder
    gc.collect()  # used to free the model first under tp_clear


def test_batch_ext_after_model_close_raises(model_path):
    model, ctx = _stack(model_path)
    batch = cy.LlamaBatchExt(ctx)
    model.close()
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        batch.add_token(1, 0)
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        batch.clear()
    del batch, ctx, model
    gc.collect()


def test_sample_with_grammar_vocab_of_closed_model_raises(model_path):
    grammar_model = cy.LlamaModel(model_path, verbose=False)
    sampler = cy.LlamaSampler()
    sampler.add_grammar(grammar_model.get_vocab(), GRAMMAR, "root")
    sampler.add_greedy()
    model, ctx = _stack(model_path)
    vocab = model.get_vocab()
    ctx.decode(cy.llama_batch_get_one(vocab.tokenize("Answer:", add_special=True, parse_special=False)))
    grammar_model.close()
    with pytest.raises(RuntimeError, match="LlamaModel is closed"):
        sampler.sample(ctx, -1)
    sampler.close()
    ctx.close()
    model.close()
    del sampler, ctx, model, vocab, grammar_model
    gc.collect()


class TestNoneAndUnboundArguments:
    """Typed arguments used to accept None and read a C field from it."""

    def test_vocab_arguments_reject_none(self):
        s = cy.LlamaSampler()
        for call in (
            lambda: s.add_grammar(None, GRAMMAR, "root"),
            lambda: s.add_infill(None),
            lambda: s.add_suppress_tokens(None),
            lambda: s.add_dry(None, 0.8, 1.75, 2, 64, []),
        ):
            with pytest.raises(TypeError):
                call()

    def test_sample_rejects_none_context(self):
        s = cy.LlamaSampler()
        s.add_greedy()
        with pytest.raises(TypeError):
            s.sample(None, -1)

    def test_closed_vocab_rejected_when_adding(self, model_path):
        model = cy.LlamaModel(model_path, verbose=False)
        vocab = model.get_vocab()
        model.close()
        with pytest.raises(RuntimeError, match="LlamaModel is closed"):
            cy.LlamaSampler().add_grammar(vocab, GRAMMAR, "root")
        del vocab, model
        gc.collect()

    def test_copy_to_closed_destination_raises(self):
        src, dst = cy.LlamaSampler(), cy.LlamaSampler()
        src.add_greedy()
        dst.add_greedy()
        dst.close()
        with pytest.raises(RuntimeError, match="closed"):
            src.copy_to(dst)

    def test_threadpool_arguments(self, model_path):
        model, ctx = _stack(model_path)
        with pytest.raises(TypeError):
            ctx.attach_threadpool(None, None)
        unbound = cy.GgmlThreadPool.__new__(cy.GgmlThreadPool)
        with pytest.raises(RuntimeError, match="GgmlThreadPool"):
            unbound.pause()
        with pytest.raises(RuntimeError, match="GgmlThreadPool"):
            cy.llama_attach_threadpool(ctx, unbound, unbound)
        ctx.close()
        model.close()
        del ctx, model
        gc.collect()

    def test_unbound_buft_override_raises(self):
        with pytest.raises(RuntimeError, match="not bound"):
            cy.LlamaModelTensorBuftOverride().pattern
