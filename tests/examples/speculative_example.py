#!/usr/bin/env python3
"""
Speculative decoding with a draft model.

A small draft model proposes tokens; the target model checks them all in one
batch and keeps the longest prefix it agrees with, plus one token of its own.
With greedy verification the output is identical to plain greedy decoding of
the target; only the speed changes. The loop follows llama.cpp's
examples/speculative-simple.

The draft model must share the target's vocabulary (same model family).

Usage:
    python speculative_example.py --target models/Qwen3-4B-Q8_0.gguf --draft models/Qwen3-0.6B-Q8_0.gguf
    python speculative_example.py --target models/Llama-3.2-1B-Instruct-Q8_0.gguf --bench 3
"""

import argparse
import time
from dataclasses import dataclass, field

from cyllama.llama.llama_cpp import (
    LlamaBatch,
    LlamaContext,
    LlamaContextParams,
    LlamaModel,
    LlamaModelParams,
    LlamaSampler,
    Speculative,
    SpeculativeParams,
    disable_logging,
)


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


def speculative_generate(
    ctx_tgt: LlamaContext, spec: Speculative, params: SpeculativeParams, prompt: list, n_predict: int
) -> Result:
    """Draft with ``spec``, verify each draft in one target batch."""
    vocab = ctx_tgt.model.get_vocab()
    sampler = _greedy()
    batch = LlamaBatch(n_tokens=max(len(prompt), params.n_max + 1), embd=0, n_seq_max=1)
    res = Result()
    t0 = time.perf_counter()

    ctx_tgt.kv_cache_clear()
    spec.begin(prompt)
    # The target processes all but the last prompt token; that token seeds the first draft.
    processed, id_last = list(prompt[:-1]), prompt[-1]
    if processed:
        batch.set_batch(processed, 0, False)
        ctx_tgt.decode(batch)

    while True:
        n_left = n_predict - len(res.tokens)
        draft = spec.draft(params, processed, id_last)[: n_left - 1] if n_left > 1 else []

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
        spec.accept(len(accepted) - 1)

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
    _, ctx_dft = load(args.draft or args.target, n_ctx, args.ngl)

    vocab = model_tgt.get_vocab()
    prompt = vocab.tokenize(args.prompt, add_special=True, parse_special=False)
    params = SpeculativeParams(n_max=args.n_max, p_min=args.p_min)
    spec = Speculative(params, ctx_tgt, ctx_dft)

    runs = max(args.bench, 1)
    if args.bench:
        greedy_generate(ctx_tgt, prompt, args.n_predict)
        speculative_generate(ctx_tgt, spec, params, prompt, args.n_predict)

    base = [greedy_generate(ctx_tgt, prompt, args.n_predict) for _ in range(runs)]
    specs = [speculative_generate(ctx_tgt, spec, params, prompt, args.n_predict) for _ in range(runs)]

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
