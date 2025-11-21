# cython: language_level=3
# ngram_cache.pxd - Cython declarations for ngram-cache.h

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp cimport bool

# Import llama token type
from llama cimport llama_token, LLAMA_TOKEN_NULL

cdef extern from "ngram-cache.h":
    # Constants
    cdef int LLAMA_NGRAM_MIN
    cdef int LLAMA_NGRAM_MAX
    cdef int LLAMA_NGRAM_STATIC

    # N-gram structure
    cdef cppclass common_ngram:
        llama_token tokens[4]  # LLAMA_NGRAM_MAX = 4
        common_ngram()
        common_ngram(const llama_token * input, const int ngram_size)
        bool operator==(const common_ngram & other)

    # Hash functions (opaque to Python)
    cdef cppclass common_token_hash_function:
        pass

    cdef cppclass common_ngram_hash_function:
        pass

    # Type aliases
    ctypedef unordered_map[llama_token, int] common_ngram_cache_part
    ctypedef unordered_map[common_ngram, common_ngram_cache_part, common_ngram_hash_function] common_ngram_cache

    # Update an ngram cache with tokens
    void common_ngram_cache_update(
        common_ngram_cache & ngram_cache,
        int ngram_min,
        int ngram_max,
        vector[llama_token] & inp_data,
        int nnew,
        bool print_progress
    ) except +

    # Try to draft tokens from ngram caches
    void common_ngram_cache_draft(
        vector[llama_token] & inp,
        vector[llama_token] & draft,
        int n_draft,
        int ngram_min,
        int ngram_max,
        common_ngram_cache & nc_context,
        common_ngram_cache & nc_dynamic,
        common_ngram_cache & nc_static
    ) except +

    # Save an ngram cache to a file
    void common_ngram_cache_save(
        common_ngram_cache & ngram_cache,
        string & filename
    ) except +

    # Load an ngram cache from a file
    common_ngram_cache common_ngram_cache_load(
        string & filename
    ) except +

    # Merge two ngram caches
    void common_ngram_cache_merge(
        common_ngram_cache & ngram_cache_target,
        common_ngram_cache & ngram_cache_add
    ) except +
