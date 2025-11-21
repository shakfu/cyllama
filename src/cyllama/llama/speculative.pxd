# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from llama cimport llama_context, llama_token
from common cimport llama_tokens

cdef extern from "speculative.h" nogil:

    # Opaque struct
    ctypedef struct common_speculative:
        pass

    # Parameters for speculative decoding
    ctypedef struct common_speculative_params:
        int n_draft      # max drafted tokens
        int n_reuse      # number of tokens to reuse from previous draft
        float p_min      # min probability required to accept a token in the draft

    # Initialize speculative decoding with target and draft contexts
    common_speculative * common_speculative_init(
        llama_context * ctx_tgt,
        llama_context * ctx_dft
    ) except +

    # Free speculative decoding resources
    void common_speculative_free(common_speculative * spec) except +

    # Check if target and draft contexts are compatible
    bool common_speculative_are_compatible(
        const llama_context * ctx_tgt,
        const llama_context * ctx_dft
    ) except +

    # Add token replacement mapping between target and draft models
    void common_speculative_add_replacement_tgt_dft(
        common_speculative * spec,
        const char * source,
        const char * dest
    ) except +

    # Generate draft tokens using the draft model
    llama_tokens common_speculative_gen_draft(
        common_speculative * spec,
        common_speculative_params params,
        const llama_tokens & prompt,
        llama_token id_last
    ) except +
