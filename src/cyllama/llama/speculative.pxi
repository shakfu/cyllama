# cython: language_level=3
# distutils: language=c++

"""
Speculative Decoding Wrappers

This module provides Cython wrappers for llama.cpp's speculative decoding functionality.
Speculative decoding can provide 2-3x inference speedup by using a smaller draft model
to generate candidate tokens that are then verified by the target model.
"""

from typing import List, Optional, Tuple
from cython.operator cimport dereference as deref
from libc.stdint cimport uint16_t
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef class SpeculativeParams:
    """Parameters for speculative decoding.

    Attributes:
        n_max: Maximum number of tokens to draft (default: 16)
        n_min: Minimum number of draft tokens (default: 0)
        p_split: Speculative decoding split probability (default: 0.1)
        p_min: Minimum probability required to accept a token (default: 0.75)
    """
    cdef common.common_params_speculative params

    def __init__(self, int n_max = 16, int n_min = 0, float p_split = 0.1, float p_min = 0.75):
        self.params.n_max = n_max
        self.params.n_min = n_min
        self.params.p_split = p_split
        self.params.p_min = p_min

    @property
    def n_max(self) -> int:
        """Maximum number of tokens to draft."""
        return self.params.n_max

    @n_max.setter
    def n_max(self, int value):
        self.params.n_max = value

    @property
    def n_min(self) -> int:
        """Minimum number of draft tokens."""
        return self.params.n_min

    @n_min.setter
    def n_min(self, int value):
        self.params.n_min = value

    @property
    def p_split(self) -> float:
        """Speculative decoding split probability."""
        return self.params.p_split

    @p_split.setter
    def p_split(self, float value):
        self.params.p_split = value

    @property
    def p_min(self) -> float:
        """Minimum probability required to accept a token in the draft."""
        return self.params.p_min

    @p_min.setter
    def p_min(self, float value):
        self.params.p_min = value

    def __repr__(self) -> str:
        return (
            f"SpeculativeParams(n_max={self.n_max}, n_min={self.n_min}, "
            f"p_split={self.p_split}, p_min={self.p_min})"
        )


cdef class Speculative:
    """Speculative decoding manager.

    This class manages speculative decoding using a target model context and
    speculative decoding parameters. The draft model generates candidate tokens
    quickly, which are then verified by the target model, potentially providing
    2-3x speedup.

    Example:
        >>> ctx_target = LlamaContext(model_target, params_target)
        >>> spec_params = SpeculativeParams(n_max=16, p_min=0.75)
        >>> spec = Speculative(spec_params, ctx_target)
        >>> draft_tokens = spec.draft(spec_params, prompt_tokens, last_token_id)
    """
    cdef speculative.common_speculative * spec
    cdef public LlamaContext ctx_tgt

    def __init__(self, SpeculativeParams params, LlamaContext ctx_target):
        """Initialize speculative decoding.

        Args:
            params: Speculative decoding parameters
            ctx_target: The target (main) model context

        Raises:
            ValueError: If the context is not compatible
            RuntimeError: If initialization fails
        """
        if not self.is_compat(ctx_target):
            raise ValueError("Target context is not compatible for speculative decoding")

        self.ctx_tgt = ctx_target
        self.spec = speculative.common_speculative_init(params.params, ctx_target.ptr)

        if self.spec == NULL:
            raise RuntimeError("Failed to initialize speculative decoding")

    def __dealloc__(self):
        if self.spec != NULL:
            speculative.common_speculative_free(self.spec)
            self.spec = NULL

    @staticmethod
    def is_compat(LlamaContext ctx_target) -> bool:
        """Check if the target context is compatible for speculative decoding.

        Args:
            ctx_target: The target model context

        Returns:
            True if compatible, False otherwise
        """
        return speculative.common_speculative_is_compat(ctx_target.ptr)

    def begin(self, list prompt_tokens):
        """Optionally call once at the beginning of a new generation.

        Args:
            prompt_tokens: List of prompt token IDs
        """
        cdef vector[llama.llama_token] prompt_vec
        cdef int token
        for token in prompt_tokens:
            prompt_vec.push_back(<llama.llama_token>token)
        speculative.common_speculative_begin(self.spec, prompt_vec)

    def draft(
        self,
        SpeculativeParams params,
        list prompt_tokens,
        int last_token_id
    ) -> List[int]:
        """Generate draft tokens using the draft model.

        Args:
            params: Speculative decoding parameters
            prompt_tokens: List of prompt token IDs
            last_token_id: ID of the last token generated

        Returns:
            List of draft token IDs
        """
        cdef vector[llama.llama_token] prompt_vec
        cdef vector[llama.llama_token] result_vec
        cdef int token

        for token in prompt_tokens:
            prompt_vec.push_back(<llama.llama_token>token)

        result_vec = speculative.common_speculative_draft(
            self.spec,
            params.params,
            prompt_vec,
            <llama.llama_token>last_token_id
        )

        cdef list result = []
        cdef size_t i
        for i in range(result_vec.size()):
            result.append(<int>result_vec[i])

        return result

    def accept(self, int n_accepted):
        """Inform the speculative decoder that n_accepted tokens were accepted.

        Args:
            n_accepted: Number of tokens accepted by the target model
        """
        speculative.common_speculative_accept(self.spec, <uint16_t>n_accepted)

    def print_stats(self):
        """Print statistics about the speculative decoding."""
        speculative.common_speculative_print_stats(self.spec)

    def __repr__(self) -> str:
        return f"Speculative(target={self.ctx_tgt})"
