# distutils: language=c++

from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t, uint64_t, uint8_t
from libc.stdio cimport FILE
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.memory cimport unique_ptr as std_unique_ptr
from libcpp.set cimport set as std_set

cimport llama
cimport common

#------------------------------------------------------------------------------
# sampling.h

cdef extern from "sampling.h":

	# common_sampler extends llama_sampler with additional functionality:
	#
	#  - grammar support
	#  - custom sampler logic based on the parameters
	#  - history of the last accepted tokens
	#  - performance metrics
	#
	# This goal is to have a common implementation of the sampling logic shared across the examples.
	# For example, depending on the temperature, the sampling chain can be very simple (greedy) or more
	# complex (top-k, top-p, etc).
	#
	# Another example is related to the grammar. In general, the grammar constraints applied on the full
	# vocabulary can be very taxing. To improve performance, the grammar can be applied only to the sampled
	# token in order to verify if it fits the grammar. And only if the token doesn't fit the grammar, the
	# grammar constraints are applied to the full vocabulary and the token is resampled.
	#
	# The common_sampler also maintains a container with the last accepted tokens. In the future, this can
	# be moved into the core llama library.
	#
	# For convenience, the common_sampler also maintains a container with the current candidate tokens.
	# This can be used to access the probabilities of the rest of the non-sampled tokens.
	#
	# TODO: measure grammar performance
	#

	ctypedef struct common_sampler: pass

	# llama_sampler API overloads

	cdef common_sampler * common_sampler_init(const llama.llama_model * model, const common.common_params_sampling & params)

	cdef void common_sampler_free(common_sampler * gsmpl)

	# if accept_grammar is true, the token is accepted both by the sampling chain and the grammar
	cdef void common_sampler_accept(common_sampler * gsmpl, llama.llama_token token, bint accept_grammar)
	cdef void common_sampler_reset (common_sampler * gsmpl)
	cdef common_sampler * common_sampler_clone (common_sampler * gsmpl)

	# arguments can be nullptr to skip printing
	cdef void common_perf_print(const llama.llama_context * ctx, const common_sampler * gsmpl)

	# extended sampling implementation:
	#
	# - set logits
	# - apply the configured sampler chain
	# - check if the token fits the grammar (if any)
	# - if not: resample by first applying the grammar constraints and then sampling again (slower path)
	#
	cdef llama.llama_token common_sampler_sample(common_sampler * gsmpl, llama.llama_context * ctx, int idx)

	# generalized version of common_sampler_sample
	#
	# will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
	# if the sampler disagrees at some point, we stop and return the accepted tokens up to now
	#
	#      common_sampler_sample_n(gsmpl, ctx, { idx }, {})
	#
	# is equivalent to
	#
	#      common_sampler_sample(gsmpl, ctx, idx)
	#      common_sampler_accept(gsmpl, token, true)
	#
	# requires: idxs.size() == draft.size() + 1
	#
	# returns at least 1 token, up to idxs.size()
	#
	cdef std_vector[llama.llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama.llama_context * ctx, const std_vector[int] & idxs, const common.llama_tokens & draft)

	# assume idxs == [ 0, 1, 2, ..., draft.size() ]
	cdef std_vector[llama.llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama.llama_context * ctx, const common.llama_tokens & draft)

	cdef uint32_t common_sampler_get_seed(const common_sampler * gsmpl)

    # -------------------------------------------------------------------------
    # helpers


	# access the internal list of current candidate tokens
	# cdef llama.llama_token_data_array * common_sampler_get_candidates(common_sampler * gsmpl)

	# get the last accepted token
	cdef llama.llama_token common_sampler_last(const common_sampler * gsmpl)

	# print the sampler chain into a string
	cdef std_string common_sampler_print(const common_sampler * gsmpl)

	# get a string representation of the last accepted tokens
	cdef std_string common_sampler_prev_str(common_sampler * gsmpl, llama.llama_context * ctx, int n)

	cdef char common_sampler_type_to_chr(common.common_sampler_type cnstr)
	cdef std_string common_sampler_type_to_str(common.common_sampler_type cnstr)

	cdef std_vector[common.common_sampler_type] common_sampler_types_from_names(const std_vector[std_string] & names, bint allow_alt_names)
	cdef std_vector[common.common_sampler_type] common_sampler_types_from_chars(const std_string & chars)

	cdef llama.llama_sampler * llama_sampler_init_llg(const llama.llama_vocab * vocab,
	                const char * grammar_kind, const char * grammar_data)

