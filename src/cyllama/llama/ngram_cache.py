"""N-gram cache for draft-model-free speculative decoding.

Pure-Python port of llama.cpp's common/ngram-cache. It uses no llama.cpp API.
Files written by ``save()`` match the format of ``llama-lookup-create``.
"""

from __future__ import annotations

import os
import struct
from typing import Optional, Sequence, Union

_NGRAM_MIN = 1
_NGRAM_MAX = 4
_NGRAM_STATIC = 2
_TOKEN_NULL = -1  # LLAMA_TOKEN_NULL


def _make_ngram(tokens: Sequence[int], size: int) -> tuple[int, ...]:
    """Create a padded n-gram tuple of length _NGRAM_MAX."""
    result = list(tokens[:size])
    while len(result) < _NGRAM_MAX:
        result.append(_TOKEN_NULL)
    return tuple(result)


class NgramCache:
    """N-gram cache for accelerating text generation with repeated patterns.

    N-gram caching stores patterns of previously generated tokens and uses them
    to predict likely continuations, speeding up generation when text contains
    repetitive patterns.

    Example:
        cache = NgramCache()
        tokens = [1, 2, 3, 4, 5, 2, 3, 4]
        cache.update(tokens, ngram_min=2, ngram_max=4)

        inp = [1, 2]
        draft = cache.draft(inp, n_draft=5, ngram_min=2, ngram_max=4)

        cache.save("cache.bin")
        cache2 = NgramCache.load("cache.bin")
        cache.merge(cache2)
    """

    def __init__(self) -> None:
        # ngram -> {next_token: count}
        self._data: dict[tuple[int, ...], dict[int, int]] = {}

    def update(
        self,
        tokens: Sequence[int],
        ngram_min: int = 2,
        ngram_max: int = 4,
        nnew: Optional[int] = None,
        print_progress: bool = False,
    ) -> None:
        """Update the n-gram cache with new tokens.

        Args:
            tokens: List of token IDs to add to the cache
            ngram_min: Minimum n-gram size (default: 2)
            ngram_max: Maximum n-gram size (default: 4, max: 4)
            nnew: Number of new tokens appended (default: len(tokens))
            print_progress: Ignored; kept for signature compatibility.
        """
        if nnew is None:
            nnew = len(tokens)

        ngram_min = max(_NGRAM_MIN, min(ngram_min, _NGRAM_MAX))
        ngram_max = max(_NGRAM_MIN, min(ngram_max, _NGRAM_MAX))

        n = len(tokens)
        data = self._data

        for ngram_size in range(ngram_min, ngram_max + 1):
            i_start = max(n - nnew, ngram_size)
            for i in range(i_start, n):
                ngram = _make_ngram(tokens[i - ngram_size : i], ngram_size)
                next_token = tokens[i]
                part = data.get(ngram)
                if part is None:
                    data[ngram] = {next_token: 1}
                else:
                    part[next_token] = part.get(next_token, 0) + 1

    def draft(
        self,
        inp: Sequence[int],
        n_draft: int = 16,
        ngram_min: int = 2,
        ngram_max: int = 4,
        context_cache: Optional[NgramCache] = None,
        dynamic_cache: Optional[NgramCache] = None,
        static_cache: Optional[NgramCache] = None,
    ) -> list[int]:
        """Draft tokens using n-gram prediction.

        Args:
            inp: Input tokens generated so far
            n_draft: Maximum number of tokens to draft (default: 16)
            ngram_min: Minimum n-gram size (default: 2)
            ngram_max: Maximum n-gram size (default: 4)
            context_cache: NgramCache based on current context (default: self)
            dynamic_cache: NgramCache based on previous generations (default: empty)
            static_cache: NgramCache from large corpus for validation (default: empty)

        Returns:
            List of drafted token IDs
        """
        ngram_min = max(_NGRAM_MIN, min(ngram_min, _NGRAM_MAX))
        ngram_max = max(_NGRAM_MIN, min(ngram_max, _NGRAM_MAX))

        ctx_data = (context_cache if context_cache is not None else self)._data
        dyn_data = (dynamic_cache if dynamic_cache is not None else NgramCache())._data
        sta_data = (static_cache if static_cache is not None else NgramCache())._data

        # Seed: last input token
        if len(inp) > 0:
            draft_tokens = [inp[-1]]
        else:
            draft_tokens = [0]

        # Threshold tables (indexed by ngram_size - 1)
        min_sample_lax = [2, 2, 1, 1]
        min_percent_lax = [66, 50, 50, 50]
        min_sample_strict = [4, 3, 2, 2]
        min_percent_strict = [75, 66, 66, 66]

        while len(draft_tokens) - 1 < n_draft:
            # Reconstruct the full sequence for lookup
            combined_seq = list(inp) + draft_tokens[1:]
            drafted = False

            # 1. Try context cache (lax thresholds)
            for ngram_size in range(ngram_max, ngram_min - 1, -1):
                idx = ngram_size - 1
                if len(combined_seq) < ngram_size:
                    continue
                ngram = _make_ngram(combined_seq[-ngram_size:], ngram_size)
                part = ctx_data.get(ngram)
                if part is None:
                    continue

                # Find best token (optionally weighted by static cache)
                best_token = _TOKEN_NULL
                best_score = -1
                sum_count = 0
                for tok, cnt in part.items():
                    sum_count += cnt
                    sta_part = (
                        sta_data.get(_make_ngram(combined_seq[-_NGRAM_STATIC:], _NGRAM_STATIC))
                        if len(combined_seq) >= _NGRAM_STATIC
                        else None
                    )
                    sta_cnt = sta_part.get(tok, 0) if sta_part else 0
                    score = cnt * max(1, sta_cnt)
                    if score > best_score:
                        best_score = score
                        best_token = tok

                if best_token == _TOKEN_NULL:
                    continue

                max_count = part.get(best_token, 0)
                if sum_count >= min_sample_lax[idx] and 100 * max_count >= min_percent_lax[idx] * sum_count:
                    draft_tokens.append(best_token)
                    drafted = True
                    break

            if drafted:
                continue

            # 2. Try dynamic cache (strict thresholds)
            for ngram_size in range(ngram_max, ngram_min - 1, -1):
                idx = ngram_size - 1
                if len(combined_seq) < ngram_size:
                    continue
                ngram = _make_ngram(combined_seq[-ngram_size:], ngram_size)
                part = dyn_data.get(ngram)
                if part is None:
                    continue

                best_token = _TOKEN_NULL
                best_score = -1
                sum_count = 0
                for tok, cnt in part.items():
                    sum_count += cnt
                    sta_part = (
                        sta_data.get(_make_ngram(combined_seq[-_NGRAM_STATIC:], _NGRAM_STATIC))
                        if len(combined_seq) >= _NGRAM_STATIC
                        else None
                    )
                    sta_cnt = sta_part.get(tok, 0) if sta_part else 0
                    score = cnt * max(1, sta_cnt)
                    if score > best_score:
                        best_score = score
                        best_token = tok

                if best_token == _TOKEN_NULL:
                    continue

                max_count = part.get(best_token, 0)
                if sum_count >= min_sample_strict[idx] and 100 * max_count >= min_percent_strict[idx] * sum_count:
                    draft_tokens.append(best_token)
                    drafted = True
                    break

            if drafted:
                continue

            # 3. Try static cache only (2-gram)
            if len(combined_seq) >= _NGRAM_STATIC:
                ngram = _make_ngram(combined_seq[-_NGRAM_STATIC:], _NGRAM_STATIC)
                part = sta_data.get(ngram)
                if part:
                    best_token = _TOKEN_NULL
                    best_count = -1
                    sum_count = 0
                    for tok, cnt in part.items():
                        sum_count += cnt
                        if cnt > best_count:
                            best_count = cnt
                            best_token = tok
                    if (
                        best_token != _TOKEN_NULL
                        and sum_count >= min_sample_lax[1]
                        and 100 * best_count >= 50 * sum_count
                    ):
                        draft_tokens.append(best_token)
                        continue

            # No source could draft
            break

        return draft_tokens[1:]  # skip seed token

    def save(self, filename: Union[str, os.PathLike[str]]) -> None:
        """Save the n-gram cache to a binary file (compatible with C++ format).

        Args:
            filename: Path where to save the cache
        """
        with open(filename, "wb") as f:
            for ngram, part in self._data.items():
                # Write 4 tokens (int32 each)
                for t in ngram:
                    f.write(struct.pack("<i", t))
                # Write number of token->count pairs
                f.write(struct.pack("<i", len(part)))
                for token, count in part.items():
                    f.write(struct.pack("<i", token))
                    f.write(struct.pack("<i", count))

    @staticmethod
    def load(filename: Union[str, os.PathLike[str]]) -> NgramCache:
        """Load an n-gram cache from a binary file.

        Args:
            filename: Path from which to load the cache

        Returns:
            NgramCache instance with loaded data
        """
        cache = NgramCache()
        ngram_bytes = _NGRAM_MAX * 4  # 4 int32s
        with open(filename, "rb") as f:
            while True:
                data = f.read(ngram_bytes)
                if len(data) < ngram_bytes:
                    break
                tokens = struct.unpack("<" + "i" * _NGRAM_MAX, data)
                ngram = tuple(tokens)
                ntokens_data = f.read(4)
                if len(ntokens_data) < 4:
                    break
                ntokens = struct.unpack("<i", ntokens_data)[0]
                part: dict[int, int] = {}
                for _ in range(ntokens):
                    entry = f.read(8)
                    if len(entry) < 8:
                        break
                    tok, cnt = struct.unpack("<ii", entry)
                    part[tok] = cnt
                cache._data[ngram] = part
        return cache

    def merge(self, other: NgramCache) -> None:
        """Merge another n-gram cache into this one.

        Args:
            other: Another NgramCache to merge into this cache
        """
        if not isinstance(other, NgramCache):
            raise TypeError("Can only merge with another NgramCache")

        for ngram, part_add in other._data.items():
            part_target = self._data.get(ngram)
            if part_target is None:
                self._data[ngram] = dict(part_add)
            else:
                for token, count in part_add.items():
                    part_target[token] = part_target.get(token, 0) + count

    def __repr__(self) -> str:
        return f"<NgramCache at {hex(id(self))}>"
