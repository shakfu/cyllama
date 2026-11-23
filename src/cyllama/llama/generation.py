"""Sampler construction, prompt decode and the sample-decode loop shared by
``LLM``, the servers, grammar-constrained generation, batch generation and the CLI.

Stop-sequence handling stays with each caller.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple

from .llama_cpp import (
    LlamaContext,
    LlamaSampler,
    LlamaSamplerChainParams,
    LlamaVocab,
    llama_batch_get_one,
)

if TYPE_CHECKING:
    from ..api import GenerationConfig


def build_sampler(
    vocab: LlamaVocab,
    config: "GenerationConfig",
    context_window: int,
    no_perf: bool = True,
    grammar: Optional[Tuple[str, str]] = None,
) -> LlamaSampler:
    """Build the sampler chain ``config`` describes.

    Args:
        vocab: Model vocabulary.
        config: Sampling parameters; the infrastructure fields are ignored.
        context_window: Tokens a ``-1`` penalty or DRY window covers (the context size).
        no_perf: Skip the chain's performance counters.
        grammar: ``(gbnf, root_rule)``; its link goes first so the other links
            choose among tokens the grammar allows.
    """
    params = LlamaSamplerChainParams()
    params.no_perf = no_perf
    sampler = LlamaSampler(params)
    if grammar is not None:
        sampler.add_grammar(vocab, grammar[0], grammar[1])
    sampler.add_suppress_tokens(vocab)

    # libllama clamps a negative penalty or DRY window to 0, which disables it;
    # GenerationConfig documents -1 as "whole context", so resolve it here.
    # Penalties and DRY shape logits before truncation and selection.
    if (
        config.repeat_penalty != 1.0 or config.frequency_penalty != 0.0 or config.presence_penalty != 0.0
    ) and config.penalty_last_n != 0:
        sampler.add_penalties(
            vocab.n_vocab,
            context_window if config.penalty_last_n < 0 else config.penalty_last_n,
            config.repeat_penalty,
            config.frequency_penalty,
            config.presence_penalty,
        )
    if config.dry_multiplier > 0.0 and config.dry_penalty_last_n != 0:
        sampler.add_dry(
            vocab,
            config.dry_multiplier,
            config.dry_base,
            config.dry_allowed_length,
            context_window if config.dry_penalty_last_n < 0 else config.dry_penalty_last_n,
            config.dry_sequence_breakers,
        )

    if config.temperature == 0.0:
        sampler.add_greedy()
    elif config.mirostat == 1:
        # Mirostat owns temperature and selection; upstream skips truncation samplers.
        sampler.add_temp(config.temperature)
        sampler.add_mirostat(vocab.n_vocab, config.seed, config.mirostat_tau, config.mirostat_eta, 100)
    elif config.mirostat == 2:
        sampler.add_temp(config.temperature)
        sampler.add_mirostat_v2(config.seed, config.mirostat_tau, config.mirostat_eta)
    elif config.top_n_sigma >= 0.0:
        # Top-n-sigma thresholds on the logits' own spread, in place of top_k/top_p/min_p.
        sampler.add_top_n_sigma(config.top_n_sigma)
        sampler.add_temp(config.temperature)
        sampler.add_dist(config.seed)
    else:
        sampler.add_min_p(config.min_p, 1)
        sampler.add_top_k(config.top_k)
        sampler.add_top_p(config.top_p, 1)
        sampler.add_temp(config.temperature)
        sampler.add_dist(config.seed)  # 0xFFFFFFFF (the default) means a random seed
    return sampler


def decode_prompt(ctx: LlamaContext, tokens: List[int], start: int = 0) -> None:
    """Decode ``tokens[start:]`` at positions ``start..``, in chunks of the context's n_batch.

    ``tokens[:start]`` must already be in sequence 0 of the context's memory.
    The last token's logits are output.
    """
    n_batch = ctx.n_batch
    for i in range(start, len(tokens), n_batch):
        ctx.decode(llama_batch_get_one(tokens[i : i + n_batch], i))


def sample_tokens(
    ctx: LlamaContext,
    sampler: LlamaSampler,
    vocab: LlamaVocab,
    n_past: int,
    max_tokens: int,
    cancelled: Optional[Callable[[], bool]] = None,
    decoded: Optional[List[int]] = None,
) -> Iterator[int]:
    """Yield up to ``max_tokens`` sampled tokens, continuing a sequence of ``n_past`` tokens.

    Each yielded token is decoded into the context before the next is sampled,
    so the consumer can stop at any point. Stops at an end-of-generation token
    (not yielded), after a token that no longer fits in the context, when
    ``cancelled()`` is true, or when the context's abort callback aborts a decode.

    ``decoded``, if given, receives each token once it is in the context's memory.
    """
    n_ctx = ctx.n_ctx
    for _ in range(max_tokens):
        if cancelled is not None and cancelled():
            return
        token = sampler.sample(ctx, -1)
        if vocab.is_eog(token):
            return
        yield token
        if n_past >= n_ctx:
            return  # no position left to decode it at
        try:
            ctx.decode(llama_batch_get_one([token], n_past))
        except InterruptedError:
            return  # cancel() aborted the decode
        n_past += 1
        if decoded is not None:
            decoded.append(token)
