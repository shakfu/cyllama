

cdef class CommonSampler:
    """cython wrapper of sampling.common_sampler"""
    cdef sampling.common_sampler * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, LlamaModel model, CommonParamsSampling params):
        self.ptr = sampling.common_sampler_init(model.ptr, params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to init Sampler")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            sampling.common_sampler_free(self.ptr)
            self.ptr = NULL

    def accept(self, llama.llama_token token, bint accept_grammar):
        """if accept_grammar is true, the token is accepted both by the sampling chain and the grammar"""
        sampling.common_sampler_accept(self.ptr, token, accept_grammar)

    def reset(self):
        """reset common sampler"""
        sampling.common_sampler_reset(self.ptr)

    def clone(self) -> CommonSampler:
        """clone sampler"""
        cdef sampling.common_sampler * smplr = sampling.common_sampler_clone(self.ptr)
        cdef CommonSampler wrapper = CommonSampler.__new__(CommonSampler)
        wrapper.ptr = smplr
        return wrapper

    def sample(self, LlamaContext ctx, int idx, bint grammar_first) -> int:
        """if grammar_first is true, the grammar is applied before the samplers (slower)

        useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
        """
        return sampling.common_sampler_sample(self.ptr, ctx.ptr, idx, grammar_first)


    def sample_and_accept_n(self, LlamaContext ctx, list[int] draft, bint grammar_first) -> list[int]:
        """generalized version of common_sampler_sample

        will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
        if the sampler disagrees at some point, we stop and return the accepted tokens up to now

          common_sampler_sample_n(gsmpl, ctx, { idx }, {});

        is equivalent to

          common_sampler_sample(gsmpl, ctx, idx);
          common_sampler_accept(gsmpl, token, true);

        requires: idxs.size() == draft.size() + 1

        returns at least 1 token, up to idxs.size()
        """
        
        cdef std_vector[int] idxs = range(len(draft))
        cdef std_vector[llama.llama_token] _draft = draft
        cdef std_vector[llama.llama_token] tokens = sampling.common_sampler_sample_and_accept_n(self.ptr, ctx.ptr, idxs, _draft, grammar_first)
        return tokens

    def get_seed(self) -> int:
        """get random seed"""
        return sampling.common_sampler_get_seed(self.ptr)

    def get_candidates(self) -> list[LlamaTokenData]:
        """access the internal list of current candidate tokens"""
        cdef llama.llama_token_data_array * arr =  sampling.common_sampler_get_candidates(self.ptr)
        _result = []
        for i in range(arr.size):
            _result.append(LlamaTokenData(arr.data[i].id, arr.data[i].logit. arr.data[i].p))
        return _result

    def get_last(self) -> int:
        """get the last accepted token"""
        return sampling.common_sampler_last(self.ptr)

    def print(self) -> str:
        """print the sampler chain into a string"""
        return sampling.common_sampler_print(self.ptr).decode()

    def prev_str(self, LlamaContext ctx, int n) -> str:
        """get a string representation of the last accepted tokens"""
        return sampling.common_sampler_prev_str(self.ptr, ctx.ptr, n).decode()

# -------------------------------------------------------------------------
# helpers

def type_to_chr(common.common_sampler_type cnstr) -> str:
    cdef int c = <int>sampling.common_sampler_type_to_chr(cnstr)
    return chr(c)

def type_to_chr(common.common_sampler_type cnstr) -> str:
    return sampling.common_sampler_type_to_str(cnstr).decode()

def types_from_names(list[str] names, bint allow_alt_names) -> list[common.common_sampler_type]:
    cdef std_vector[std_string] _names = names
    cdef std_vector[common.common_sampler_type] vec = sampling.common_sampler_types_from_names(_names, allow_alt_names)
    return vec

def types_from_chars(str chars) -> str:
    cdef std_string _chars = chars
    return sampling.common_sampler_types_from_chars(_chars)

