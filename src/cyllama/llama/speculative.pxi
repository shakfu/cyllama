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
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef class SpeculativeParams:
    """
    Parameters for speculative decoding.

    Attributes:
        n_draft: Maximum number of tokens to draft (default: 16)
        n_reuse: Number of tokens to reuse from previous draft (default: 256)
        p_min: Minimum probability required to accept a token (default: 0.75)
    """
    cdef speculative.common_speculative_params params

    def __init__(self, int n_draft = 16, int n_reuse = 256, float p_min = 0.75):
        """
        Initialize speculative decoding parameters.

        Args:
            n_draft: Maximum number of tokens to draft (default: 16)
            n_reuse: Number of tokens to reuse from previous draft (default: 256)
            p_min: Minimum probability required to accept a token (default: 0.75)
        """
        self.params.n_draft = n_draft
        self.params.n_reuse = n_reuse
        self.params.p_min = p_min

    @property
    def n_draft(self) -> int:
        """Maximum number of tokens to draft."""
        return self.params.n_draft

    @n_draft.setter
    def n_draft(self, int value):
        self.params.n_draft = value

    @property
    def n_reuse(self) -> int:
        """Number of tokens to reuse from previous draft."""
        return self.params.n_reuse

    @n_reuse.setter
    def n_reuse(self, int value):
        self.params.n_reuse = value

    @property
    def p_min(self) -> float:
        """Minimum probability required to accept a token in the draft."""
        return self.params.p_min

    @p_min.setter
    def p_min(self, float value):
        self.params.p_min = value

    def __repr__(self) -> str:
        return f"SpeculativeParams(n_draft={self.n_draft}, n_reuse={self.n_reuse}, p_min={self.p_min})"


cdef class Speculative:
    """
    Speculative decoding manager.

    This class manages speculative decoding using a target model and a smaller draft model.
    The draft model generates candidate tokens quickly, which are then verified by the
    target model, potentially providing 2-3x speedup.

    Example:
        >>> # Create contexts for target and draft models
        >>> ctx_target = LlamaContext(model_target, params_target)
        >>> ctx_draft = LlamaContext(model_draft, params_draft)
        >>>
        >>> # Initialize speculative decoding
        >>> spec = Speculative(ctx_target, ctx_draft)
        >>>
        >>> # Generate draft tokens
        >>> params = SpeculativeParams(n_draft=16, p_min=0.75)
        >>> draft_tokens = spec.gen_draft(params, prompt_tokens, last_token_id)
    """
    cdef speculative.common_speculative * spec
    cdef public LlamaContext ctx_tgt
    cdef public LlamaContext ctx_dft

    def __init__(self, LlamaContext ctx_target, LlamaContext ctx_draft):
        """
        Initialize speculative decoding with target and draft contexts.

        Args:
            ctx_target: The target (main) model context
            ctx_draft: The draft (smaller, faster) model context

        Raises:
            ValueError: If the contexts are not compatible
            RuntimeError: If initialization fails
        """
        if not self.are_compatible(ctx_target, ctx_draft):
            raise ValueError("Target and draft contexts are not compatible for speculative decoding")

        self.ctx_tgt = ctx_target
        self.ctx_dft = ctx_draft
        self.spec = speculative.common_speculative_init(ctx_target.ptr, ctx_draft.ptr)

        if self.spec == NULL:
            raise RuntimeError("Failed to initialize speculative decoding")

    def __dealloc__(self):
        """Clean up speculative decoding resources."""
        if self.spec != NULL:
            speculative.common_speculative_free(self.spec)
            self.spec = NULL

    @staticmethod
    def are_compatible(LlamaContext ctx_target, LlamaContext ctx_draft) -> bool:
        """
        Check if target and draft contexts are compatible for speculative decoding.

        Args:
            ctx_target: The target model context
            ctx_draft: The draft model context

        Returns:
            True if compatible, False otherwise
        """
        return speculative.common_speculative_are_compatible(ctx_target.ptr, ctx_draft.ptr)

    def add_replacement(self, str source, str dest):
        """
        Add token replacement mapping between target and draft models.

        This is useful when the target and draft models have different tokenizers.

        Args:
            source: Source token string (from target model)
            dest: Destination token string (for draft model)
        """
        cdef bytes source_bytes = source.encode('utf-8')
        cdef bytes dest_bytes = dest.encode('utf-8')
        speculative.common_speculative_add_replacement_tgt_dft(
            self.spec,
            source_bytes,
            dest_bytes
        )

    def gen_draft(
        self,
        SpeculativeParams params,
        list prompt_tokens,
        int last_token_id
    ) -> List[int]:
        """
        Generate draft tokens using the draft model.

        Args:
            params: Speculative decoding parameters
            prompt_tokens: List of prompt token IDs
            last_token_id: ID of the last token generated

        Returns:
            List of draft token IDs

        Example:
            >>> params = SpeculativeParams(n_draft=16, p_min=0.75)
            >>> draft = spec.gen_draft(params, [1, 2, 3, 4], last_token=5)
        """
        cdef vector[llama.llama_token] prompt_vec
        cdef vector[llama.llama_token] result_vec
        cdef int token

        # Convert Python list to C++ vector
        for token in prompt_tokens:
            prompt_vec.push_back(<llama.llama_token>token)

        # Generate draft tokens
        result_vec = speculative.common_speculative_gen_draft(
            self.spec,
            params.params,
            prompt_vec,
            <llama.llama_token>last_token_id
        )

        # Convert result back to Python list
        cdef list result = []
        cdef size_t i
        for i in range(result_vec.size()):
            result.append(<int>result_vec[i])

        return result

    def __repr__(self) -> str:
        return f"Speculative(target={self.ctx_tgt}, draft={self.ctx_dft})"
