

# cdef class CommonSampler:
#     """cython wrapper of llama_cpp.common_sampler"""
#     cdef llama_cpp.common_sampler * ptr
#     cdef bint owner

#     def __cinit__(self):
#         self.ptr = NULL
#         self.owner = True

#     def __init__(self, LlamaModel model, CommonParamsSampling params):
#         self.ptr = llama_cpp.common_sampler_init(model.ptr, params.p)

#         if self.ptr is NULL:
#             raise ValueError("Failed to init Sampler")

#     def __dealloc__(self):
#         if self.ptr is not NULL and self.owner is True:
#             llama_cpp.common_sampler_free(self.ptr)
#             self.ptr = NULL

#     def accept(self, llama_cpp.llama_token token, bint accept_grammar):
#         """if accept_grammar is true, the token is accepted both by the sampling chain and the grammar"""
#         llama_cpp.common_sampler_accept(self.ptr, token, accept_grammar)

#     def reset(self):
#         """reset common sampler"""
#         llama_cpp.common_sampler_reset(self.ptr)

#     def clone(self) -> CommonSampler:
#         """clone sampler"""
#         cdef llama_cpp.common_sampler * smplr = llama_cpp.common_sampler_clone(self.ptr)
#         cdef CommonSampler wrapper = CommonSampler.__new__(CommonSampler)
#         wrapper.ptr = smplr
#         return wrapper

#     def sample(self, LlamaContext ctx, int idx, bint grammar_first) -> int:
#         """if grammar_first is true, the grammar is applied before the samplers (slower)

#         useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
#         """
#         return llama_cpp.common_sampler_sample(self.ptr, ctx.ptr, idx, grammar_first)


#     # generalized version of common_sampler_sample
#     #
#     # will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
#     # if the sampler disagrees at some point, we stop and return the accepted tokens up to now
#     #
#     #      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
#     #
#     # is equivalent to
#     #
#     #      common_sampler_sample(gsmpl, ctx, idx);
#     #      common_sampler_accept(gsmpl, token, true);
#     #
#     # requires: idxs.size() == draft.size() + 1
#     #
#     # returns at least 1 token, up to idxs.size()

#     # std_vector[llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama_context * ctx, const std_vector[int] & idxs, const llama_tokens & draft, bint grammar_first)

#     # assume idxs == [ 0, 1, 2, ..., draft.size() ]
#     # std_vector[llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama_context * ctx, const llama_tokens & draft, bint grammar_first)


#     def get_seed(self) -> int:
#         """get random seed"""
#         return llama_cpp.common_sampler_get_seed(self.ptr)

#     # def get_candidates(self) -> LlamaTokenDataArray:
#     #     """access the internal list of current candidate tokens"""
#     #     return llama_cpp.common_sampler_get_candidates(self.ptr)

#     def get_last(self) -> int:
#         """get the last accepted token"""
#         return llama_cpp.common_sampler_last(self.ptr)

#     def print(self) -> str:
#         """print the sampler chain into a string"""
#         return llama_cpp.common_sampler_print(self.ptr).decode()

#     def prev_str(self, LlamaContext ctx, int n) -> str:
#         """get a string representation of the last accepted tokens"""
#         return llama_cpp.common_sampler_prev_str(self.ptr, ctx.ptr, n).decode()

#     # char common_sampler_type_to_chr(common_sampler_type cnstr)
#     # std_string common_sampler_type_to_str(common_sampler_type cnstr)

#     # std_vector[common_sampler_type] common_sampler_types_from_names(const std_vector[std_string] & names, bint allow_alt_names)
#     # std_vector[common_sampler_type] common_sampler_types_from_chars(const std_string & chars)


