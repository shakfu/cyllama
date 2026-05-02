# cython: language_level=3
# distutils: language=c++

"""
Speculative Decoding (pure Python/Cython implementation using public API)

Provides speculative decoding using a draft model to generate candidate tokens
that are verified by the target model, potentially providing 2-3x speedup.
"""

from typing import List, Optional


class SpeculativeParams:
    """Parameters for speculative decoding.

    Attributes:
        n_max: Maximum number of tokens to draft (default: 16)
        n_min: Minimum number of draft tokens (default: 0)
        p_split: Speculative decoding split probability (default: 0.1)
        p_min: Minimum probability required to accept a token (default: 0.75)
    """

    def __init__(self, int n_max=16, int n_min=0, float p_split=0.1, float p_min=0.75):
        self.n_max = n_max
        self.n_min = n_min
        self.p_split = p_split
        self.p_min = p_min

    def __repr__(self) -> str:
        return (
            f"SpeculativeParams(n_max={self.n_max}, n_min={self.n_min}, "
            f"p_split={self.p_split}, p_min={self.p_min})"
        )


cdef class Speculative:
    """Speculative decoding manager using the public llama API.

    Uses a draft model context to generate candidate tokens quickly, which are
    then verified by the target model. Manages KV cache state for both models.

    Example:
        >>> ctx_target = LlamaContext(model_target, params_target)
        >>> ctx_draft = LlamaContext(model_draft, params_draft)
        >>> spec_params = SpeculativeParams(n_max=16, p_min=0.75)
        >>> spec = Speculative(spec_params, ctx_target, ctx_draft)
        >>> draft_tokens = spec.draft(spec_params, prompt_tokens, last_token_id)
    """
    cdef public LlamaContext ctx_tgt
    cdef public LlamaContext ctx_dft
    cdef public LlamaSampler sampler
    cdef list _draft_prompt   # tracks what's in the draft KV cache
    cdef int _n_acc_drafts
    cdef int _n_acc_tokens
    cdef int _n_gen_drafts
    cdef int _n_gen_tokens

    def __init__(self, params, LlamaContext ctx_target, LlamaContext ctx_draft=None):
        """Initialize speculative decoding.

        Args:
            params: SpeculativeParams instance
            ctx_target: The target (main) model context
            ctx_draft: The draft (smaller) model context. Required for draft-model
                       speculative decoding.

        Raises:
            ValueError: If the context is not compatible
            RuntimeError: If initialization fails (e.g., no draft context provided)
        """
        if not self.is_compat(ctx_target):
            raise ValueError("Target context is not compatible for speculative decoding")

        if ctx_draft is None:
            raise RuntimeError("Failed to initialize speculative decoding: no draft context provided")

        self.ctx_tgt = ctx_target
        self.ctx_dft = ctx_draft
        self._draft_prompt = []
        self._n_acc_drafts = 0
        self._n_acc_tokens = 0
        self._n_gen_drafts = 0
        self._n_gen_tokens = 0

        # Create a sampler for the draft model: top-k=10 + dist
        cdef LlamaSamplerChainParams sparams = LlamaSamplerChainParams()
        sparams.no_perf = True
        self.sampler = LlamaSampler(sparams)
        self.sampler.add_top_k(10)
        self.sampler.add_dist(0)

    @staticmethod
    def is_compat(LlamaContext ctx_target) -> bool:
        """Check if the target context's model supports speculative decoding.

        Tests that a context built from the same model supports partial KV
        cache removal, which is required for the draft-verify loop.

        The probe runs on a small scratch context built from the same model,
        not on `ctx_target` itself, so any KV state the caller has built up
        in `ctx_target` is preserved. Whether partial KV removal is supported
        is a property of the model architecture (recurrent vs. attention) and
        memory backend defaults, not of the caller's specific params, so a
        scratch context with default params gives the right answer.
        """
        if ctx_target is None or ctx_target.model is None:
            return False
        if ctx_target.model.ptr is NULL:
            return False

        # Build a minimal scratch context from the same model. Keep n_ctx /
        # n_batch / n_ubatch / n_seq_max small so the probe is cheap.
        cdef llama.llama_context_params scratch_params = (
            llama.llama_context_default_params()
        )
        scratch_params.n_ctx = 4
        scratch_params.n_batch = 4
        scratch_params.n_ubatch = 4
        scratch_params.n_seq_max = 1

        cdef llama.llama_context * scratch = llama.llama_init_from_model(
            ctx_target.model.ptr, scratch_params
        )
        if scratch is NULL:
            return False

        cdef llama.llama_memory_t mem
        cdef llama.llama_batch batch
        cdef int rc
        cdef bint can_rm = False
        try:
            mem = llama.llama_get_memory(scratch)
            if mem is NULL:
                return False

            # Test: decode 2 dummy tokens, then try partial removal
            batch = llama.llama_batch_init(2, 0, 1)
            batch.n_tokens = 2
            batch.token[0] = 0
            batch.token[1] = 0
            batch.pos[0] = 0
            batch.pos[1] = 1
            batch.n_seq_id[0] = 1
            batch.n_seq_id[1] = 1
            batch.seq_id[0][0] = 0
            batch.seq_id[1][0] = 0
            batch.logits[0] = False
            batch.logits[1] = False

            rc = llama.llama_decode(scratch, batch)
            llama.llama_batch_free(batch)

            if rc != 0:
                return False

            # Test partial removal (position 1 to end)
            can_rm = llama.llama_memory_seq_rm(mem, 0, 1, -1)
            llama.llama_synchronize(scratch)
            return can_rm
        finally:
            llama.llama_free(scratch)

    def begin(self, list prompt_tokens):
        """Optionally call at the beginning of a new generation to reset draft state.

        Args:
            prompt_tokens: List of prompt token IDs
        """
        self._draft_prompt = []

    def draft(self, params, list prompt_tokens, int last_token_id) -> list:
        """Generate draft tokens using the draft model.

        Args:
            params: SpeculativeParams instance
            prompt_tokens: List of prompt token IDs (in target model's vocabulary)
            last_token_id: ID of the last accepted token

        Returns:
            List of draft token IDs
        """
        cdef int n_max = params.n_max
        cdef float p_min = params.p_min
        cdef int n_ctx = llama.llama_n_ctx(self.ctx_dft.ptr) - n_max
        cdef list prompt
        cdef int reuse_n = 0
        cdef list old_prompt
        cdef int min_len
        cdef llama.llama_memory_t mem
        cdef int i_start, n_new, n_past, i
        cdef llama.llama_batch batch
        cdef llama.llama_token sampled
        cdef int32_t rc
        cdef list result = []
        cdef llama.llama_context * ctx_ptr = self.ctx_dft.ptr

        if n_ctx <= 0:
            return []

        # Trim prompt to fit in draft context
        prompt = prompt_tokens
        if len(prompt) > n_ctx:
            prompt = prompt[-n_ctx:]

        # KV cache reuse: find how much of the old draft prompt matches
        old_prompt = self._draft_prompt
        min_len = min(len(old_prompt), len(prompt))
        for i in range(min_len):
            if old_prompt[i] == prompt[i]:
                reuse_n += 1
            else:
                break

        # Manage KV cache
        mem = llama.llama_get_memory(self.ctx_dft.ptr)
        if reuse_n == 0:
            if mem is not NULL:
                llama.llama_memory_clear(mem, True)
        else:
            if reuse_n < len(old_prompt) and mem is not NULL:
                llama.llama_memory_seq_rm(mem, 0, <llama.llama_pos>reuse_n, <llama.llama_pos>-1)

        # Encode new prompt tokens that aren't in cache
        i_start = reuse_n
        n_new = len(prompt) - i_start
        if n_new > 0:
            batch = llama.llama_batch_init(n_new, 0, 1)
            batch.n_tokens = n_new
            for i in range(n_new):
                batch.token[i] = <llama.llama_token>prompt[i_start + i]
                batch.pos[i] = <llama.llama_pos>(i_start + i)
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = (i == n_new - 1)
            with nogil:
                rc = llama.llama_decode(ctx_ptr, batch)
            llama.llama_batch_free(batch)
            if rc != 0:
                raise RuntimeError(
                    f"speculative draft prompt decode failed (llama_decode rc={rc})"
                )

        self._draft_prompt = list(prompt)

        # Draft generation loop
        n_past = len(prompt)
        llama.llama_sampler_reset(self.sampler.ptr)

        for i in range(n_max):
            sampled = llama.llama_sampler_sample(self.sampler.ptr, self.ctx_dft.ptr, -1)
            llama.llama_sampler_accept(self.sampler.ptr, sampled)
            result.append(<int>sampled)

            if len(result) >= n_max:
                break

            # Decode the sampled token for next iteration
            batch = llama.llama_batch_init(1, 0, 1)
            batch.n_tokens = 1
            batch.token[0] = sampled
            batch.pos[0] = <llama.llama_pos>(n_past + i)
            batch.n_seq_id[0] = 1
            batch.seq_id[0][0] = 0
            batch.logits[0] = True
            with nogil:
                rc = llama.llama_decode(ctx_ptr, batch)
            llama.llama_batch_free(batch)
            if rc != 0:
                # Surface failure rather than continuing with stale logits
                # in the next sampler iteration.
                raise RuntimeError(
                    f"speculative draft step decode failed (llama_decode rc={rc})"
                )

            self._draft_prompt.append(<int>sampled)

        if len(result) > 0:
            self._n_gen_drafts += 1
            self._n_gen_tokens += len(result)

        return result

    def accept(self, int n_accepted):
        """Inform the speculative decoder that n_accepted tokens were accepted.

        Args:
            n_accepted: Number of tokens accepted by the target model
        """
        if n_accepted > 0:
            self._n_acc_drafts += 1
            self._n_acc_tokens += n_accepted

    def print_stats(self):
        """Print statistics about the speculative decoding."""
        import sys
        acc_rate = 0.0
        if self._n_gen_tokens > 0:
            acc_rate = 100.0 * self._n_acc_tokens / self._n_gen_tokens
        print(
            f"speculative: gen_drafts={self._n_gen_drafts}, "
            f"acc_drafts={self._n_acc_drafts}, "
            f"gen_tokens={self._n_gen_tokens}, "
            f"acc_tokens={self._n_acc_tokens}, "
            f"acc_rate={acc_rate:.1f}%",
            file=sys.stderr
        )

    def __repr__(self) -> str:
        return f"Speculative(target={self.ctx_tgt})"
