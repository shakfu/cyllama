# cython: language_level=3
# distutils: language=c++

from libcpp cimport bool
from libc.stdint cimport uint16_t
from llama cimport llama_context, llama_token
from common cimport llama_tokens, common_params_speculative

cdef extern from "speculative.h" nogil:

    # Opaque struct
    ctypedef struct common_speculative:
        pass

    # Check if the llama_context is compatible for speculative decoding
    bool common_speculative_is_compat(llama_context * ctx_tgt) except +

    # Initialize speculative decoding
    common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context * ctx_tgt
    ) except +

    # Free speculative decoding resources
    void common_speculative_free(common_speculative * spec) except +

    # Optionally call once at the beginning of a new generation
    void common_speculative_begin(
        common_speculative * spec,
        const llama_tokens & prompt
    ) except +

    # Sample up to n_draft tokens and add them to the batch using the draft model
    llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt,
        llama_token id_last
    ) except +

    # Informs the speculative decoder that n_accepted tokens were accepted
    void common_speculative_accept(
        common_speculative * spec,
        uint16_t n_accepted
    ) except +

    # Print statistics about the speculative decoding
    void common_speculative_print_stats(const common_speculative * spec) except +
