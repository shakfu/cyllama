#!/usr/bin/env python3
"""
Speculative decoding with a draft model.

A drafter proposes tokens; the target model checks them all in one batch and
keeps the longest prefix it agrees with, plus one token of its own. Two
drafters are shown: a small draft model (``Drafter``) and an n-gram lookup
over the tokens seen so far (``NgramDrafter``), which needs no second model.
With greedy verification the output is identical to plain greedy decoding of
the target; only the speed changes. The loop follows llama.cpp's
examples/speculative-simple, built on the public API only (no libcommon).

The draft model must share the target's vocabulary (same model family).
N-gram drafting pays off on repetitive output (code, lists, quoting the prompt).

Usage:
    python speculative_example.py --target models/Qwen3-4B-Q8_0.gguf --draft models/Qwen3-0.6B-Q8_0.gguf
    python speculative_example.py --target models/Llama-3.2-1B-Instruct-Q8_0.gguf --bench 3
    python speculative_example.py --target models/Llama-3.2-1B-Instruct-Q8_0.gguf --ngram
"""

import argparse
import heapq
import math
import time
from dataclasses import dataclass, field

from cyllama.llama.llama_cpp import (
    LlamaBatch,
    LlamaBatchExt,
    LlamaContext,
    LlamaContextParams,
    LlamaModel,
    LlamaModelParams,
    LlamaSampler,
    disable_logging,
)
from cyllama.llama.ngram_cache import NgramCache

DRAFT_TOP_K = 10


@dataclass
class Result:
    tokens: list = field(default_factory=list)
    seconds: float = 0.0
    n_drafted: int = 0
    n_accepted: int = 0
    n_rounds: int = 0

    @property
    def tokens_per_second(self) -> float:
        return len(self.tokens) / self.seconds if self.seconds else 0.0

    @property
    def acceptance(self) -> float:
        return self.n_accepted / self.n_drafted if self.n_drafted else 0.0


def load(path: str, n_ctx: int, n_gpu_layers: int):
    mparams = LlamaModelParams()
    mparams.n_gpu_layers = n_gpu_layers
    model = LlamaModel(path, mparams, verbose=False)
    cparams = LlamaContextParams()
    cparams.n_ctx = n_ctx
    return model, LlamaContext(model, cparams, verbose=False)


def _greedy() -> LlamaSampler:
    s = LlamaSampler()
    s.add_greedy()
    return s


def greedy_generate(ctx: LlamaContext, prompt: list, n_predict: int) -> Result:
    """Baseline: one target decode per generated token."""
    vocab = ctx.model.get_vocab()
    sampler = _greedy()
    batch = LlamaBatch(n_tokens=len(prompt), embd=0, n_seq_max=1)
    res = Result()
    t0 = time.perf_counter()

    ctx.kv_cache_clear()
    batch.set_batch(prompt, 0, False)
    ctx.decode(batch)
    n_past = len(prompt)
    while len(res.tokens) < n_predict:
        tok = sampler.sample(ctx, -1)
        res.tokens.append(tok)
        if vocab.is_eog(tok):
            break
        batch.set_batch([tok], n_past, False)
        ctx.decode(batch)
        n_past += 1

    res.seconds = time.perf_counter() - t0
    return res


class Drafter:
    """Draft-model speculator, as upstream's draft-simple with greedy drafting.

    Extends the target's newest token with the draft model's top candidate.
    Stops at ``n_max`` tokens, at an end-of-generation token, or when the top
    candidate's probability (softmax over the top 10 logits) falls below
    ``p_min``. A draft shorter than ``n_min`` is discarded. The draft KV cache
    keeps the longest prefix shared with the previous call.
    """

    def __init__(self, ctx: LlamaContext, n_max: int = 3, n_min: int = 0, p_min: float = 0.0):
        self.ctx, self.n_max, self.n_min, self.p_min = ctx, n_max, n_min, p_min
        self.vocab = ctx.model.get_vocab()
        self.sampler = LlamaSampler()
        self.sampler.add_top_k(DRAFT_TOP_K)
        self.sampler.add_greedy()
        self.cached = []  # tokens in the draft KV cache

    def _decode(self, tokens: list, pos0: int, logits_last: bool):
        batch = LlamaBatchExt(self.ctx)
        for i, tok in enumerate(tokens):
            batch.add_token(tok, pos0 + i, output=logits_last and i == len(tokens) - 1)
        self.ctx.process(batch)

    def _top_p(self) -> float:
        top = heapq.nlargest(DRAFT_TOP_K, self.ctx.get_logits_ith(-1))
        return 1.0 / sum(math.exp(x - top[0]) for x in top)

    def draft(self, processed: list, id_last: int) -> list:
        """Return draft tokens predicted to follow ``processed + [id_last]``."""
        n_ctx = self.ctx.n_ctx - self.n_max
        prompt = processed[-n_ctx:] if len(processed) > n_ctx else processed

        reuse = 0
        for a, b in zip(self.cached, prompt):
            if a != b:
                break
            reuse += 1
        if reuse == 0:
            self.ctx.kv_cache_clear()
        elif reuse < len(self.cached):
            self.ctx.memory_seq_rm(0, reuse, -1)
        if len(prompt) > reuse:
            self._decode(prompt[reuse:], reuse, False)

        n_past = len(prompt)
        self._decode([id_last], n_past, True)
        self.cached = list(prompt) + [id_last]
        self.sampler.reset()

        result = []
        while len(result) < self.n_max:
            if self.p_min > 0.0 and self._top_p() < self.p_min:
                break
            tok = self.sampler.sample(self.ctx, -1)
            result.append(tok)
            if self.vocab.is_eog(tok) or len(result) == self.n_max:
                break
            self._decode([tok], n_past + len(result), True)
            self.cached.append(tok)

        return result if len(result) >= self.n_min else []


class NgramDrafter:
    """Draft-model-free speculator: n-gram lookup over the tokens seen so far.

    Mirrors llama.cpp's lookup decoding with only the context cache. The cache
    is reset when ``processed`` no longer extends the tokens it has seen.
    """

    def __init__(self, n_max: int = 8, ngram_min: int = 1, ngram_max: int = 4):
        self.n_max, self.ngram_min, self.ngram_max = n_max, ngram_min, ngram_max
        self.cache = NgramCache()
        self.seen = []

    def draft(self, processed: list, id_last: int) -> list:
        """Return draft tokens predicted to follow ``processed + [id_last]``."""
        tokens = list(processed) + [id_last]
        if tokens[: len(self.seen)] != self.seen:
            self.cache, self.seen = NgramCache(), []
        self.cache.update(tokens, self.ngram_min, self.ngram_max, nnew=len(tokens) - len(self.seen))
        self.seen = tokens
        return self.cache.draft(tokens, n_draft=self.n_max, ngram_min=self.ngram_min, ngram_max=self.ngram_max)


def speculative_generate(ctx_tgt: LlamaContext, drafter, prompt: list, n_predict: int) -> Result:
    """Draft with ``drafter``, verify each draft in one target batch."""
    vocab = ctx_tgt.model.get_vocab()
    sampler = _greedy()
    batch = LlamaBatch(n_tokens=max(len(prompt), drafter.n_max + 1), embd=0, n_seq_max=1)
    res = Result()
    t0 = time.perf_counter()

    ctx_tgt.kv_cache_clear()
    # The target processes all but the last prompt token; that token seeds the first draft.
    processed, id_last = list(prompt[:-1]), prompt[-1]
    if processed:
        batch.set_batch(processed, 0, False)
        ctx_tgt.decode(batch)

    while True:
        n_left = n_predict - len(res.tokens)
        draft = drafter.draft(processed, id_last)[: n_left - 1] if n_left > 1 else []

        # Score id_last and every draft token in one batch; row i predicts position i + 1.
        batch.set_batch([id_last] + draft, len(processed), True)
        ctx_tgt.decode(batch)

        # Keep the draft prefix the target agrees with, plus the target's own next token.
        accepted = []
        for i in range(len(draft) + 1):
            tok = sampler.sample(ctx_tgt, i)
            accepted.append(tok)
            if i == len(draft) or tok != draft[i]:
                break

        res.n_rounds += 1
        res.n_drafted += len(draft)
        res.n_accepted += len(accepted) - 1

        processed += [id_last] + accepted[:-1]
        id_last = accepted[-1]
        # Drop the rejected draft tokens from the target's KV cache.
        ctx_tgt.memory_seq_rm(0, len(processed), -1)

        for tok in accepted:
            res.tokens.append(tok)
            if vocab.is_eog(tok) or len(res.tokens) >= n_predict:
                res.seconds = time.perf_counter() - t0
                return res


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, help="target model (GGUF)")
    parser.add_argument("--draft", help="draft model (GGUF); defaults to the target itself")
    parser.add_argument("--ngram", action="store_true", help="draft by n-gram lookup instead of a draft model")
    parser.add_argument("-p", "--prompt", default="Write a short story about a lighthouse keeper.")
    parser.add_argument("-n", "--n-predict", type=int, default=128)
    parser.add_argument("--n-max", type=int, default=8, help="max draft tokens per round")
    parser.add_argument("--p-min", type=float, default=0.75, help="stop drafting below this top-token probability")
    parser.add_argument("--ngl", type=int, default=99, help="GPU layers for both models")
    parser.add_argument("--bench", type=int, default=0, metavar="N", help="time N runs of each mode after a warm-up")
    args = parser.parse_args()

    disable_logging()
    n_ctx = 4096
    model_tgt, ctx_tgt = load(args.target, n_ctx, args.ngl)

    vocab = model_tgt.get_vocab()
    prompt = vocab.tokenize(args.prompt, add_special=True, parse_special=False)
    if args.ngram:
        drafter = NgramDrafter(n_max=args.n_max)
    else:
        _, ctx_dft = load(args.draft or args.target, n_ctx, args.ngl)
        drafter = Drafter(ctx_dft, n_max=args.n_max, p_min=args.p_min)

    runs = max(args.bench, 1)
    if args.bench:
        greedy_generate(ctx_tgt, prompt, args.n_predict)
        speculative_generate(ctx_tgt, drafter, prompt, args.n_predict)

    base = [greedy_generate(ctx_tgt, prompt, args.n_predict) for _ in range(runs)]
    specs = [speculative_generate(ctx_tgt, drafter, prompt, args.n_predict) for _ in range(runs)]

    print("".join(vocab.token_to_piece(t, 0, False) for t in specs[-1].tokens))
    print()
    b = min(base, key=lambda r: r.seconds)
    s = min(specs, key=lambda r: r.seconds)
    print(f"baseline:    {len(b.tokens)} tokens, {b.tokens_per_second:7.1f} tok/s")
    print(
        f"speculative: {len(s.tokens)} tokens, {s.tokens_per_second:7.1f} tok/s, "
        f"acceptance {s.acceptance:.1%} ({s.n_accepted}/{s.n_drafted}), {s.n_rounds} rounds"
    )
    print(f"speedup:     {s.tokens_per_second / b.tokens_per_second:.2f}x (best of {runs})")
    print(f"identical:   {s.tokens == b.tokens}")


if __name__ == "__main__":
    main()
