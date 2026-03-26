
from libc.math cimport INFINITY
from collections import deque


# -------------------------------------------------------------------------
# Sampler type helpers (pure Python, replacing sampling.h C++ functions)

_SAMPLER_TYPE_TO_CHR = {
    common.COMMON_SAMPLER_TYPE_DRY:         'd',
    common.COMMON_SAMPLER_TYPE_TOP_K:       'k',
    common.COMMON_SAMPLER_TYPE_TYPICAL_P:   'y',
    common.COMMON_SAMPLER_TYPE_TOP_P:       'p',
    common.COMMON_SAMPLER_TYPE_TOP_N_SIGMA: 's',
    common.COMMON_SAMPLER_TYPE_MIN_P:       'm',
    common.COMMON_SAMPLER_TYPE_TEMPERATURE: 't',
    common.COMMON_SAMPLER_TYPE_XTC:         'x',
    common.COMMON_SAMPLER_TYPE_INFILL:      'i',
    common.COMMON_SAMPLER_TYPE_PENALTIES:   'e',
    common.COMMON_SAMPLER_TYPE_ADAPTIVE_P:  'a',
}

_SAMPLER_TYPE_TO_STR = {
    common.COMMON_SAMPLER_TYPE_DRY:         'dry',
    common.COMMON_SAMPLER_TYPE_TOP_K:       'top_k',
    common.COMMON_SAMPLER_TYPE_TYPICAL_P:   'typ_p',
    common.COMMON_SAMPLER_TYPE_TOP_P:       'top_p',
    common.COMMON_SAMPLER_TYPE_TOP_N_SIGMA: 'top_n_sigma',
    common.COMMON_SAMPLER_TYPE_MIN_P:       'min_p',
    common.COMMON_SAMPLER_TYPE_TEMPERATURE: 'temperature',
    common.COMMON_SAMPLER_TYPE_XTC:         'xtc',
    common.COMMON_SAMPLER_TYPE_INFILL:      'infill',
    common.COMMON_SAMPLER_TYPE_PENALTIES:   'penalties',
    common.COMMON_SAMPLER_TYPE_ADAPTIVE_P:  'adaptive_p',
}

_SAMPLER_NAME_TO_TYPE = {
    'dry':         common.COMMON_SAMPLER_TYPE_DRY,
    'top_k':       common.COMMON_SAMPLER_TYPE_TOP_K,
    'top_p':       common.COMMON_SAMPLER_TYPE_TOP_P,
    'top_n_sigma': common.COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
    'typ_p':       common.COMMON_SAMPLER_TYPE_TYPICAL_P,
    'min_p':       common.COMMON_SAMPLER_TYPE_MIN_P,
    'temperature': common.COMMON_SAMPLER_TYPE_TEMPERATURE,
    'xtc':         common.COMMON_SAMPLER_TYPE_XTC,
    'infill':      common.COMMON_SAMPLER_TYPE_INFILL,
    'penalties':   common.COMMON_SAMPLER_TYPE_PENALTIES,
    'adaptive_p':  common.COMMON_SAMPLER_TYPE_ADAPTIVE_P,
}

_SAMPLER_ALT_NAME_TO_TYPE = {
    'top-k':       common.COMMON_SAMPLER_TYPE_TOP_K,
    'top-p':       common.COMMON_SAMPLER_TYPE_TOP_P,
    'top-n-sigma': common.COMMON_SAMPLER_TYPE_TOP_N_SIGMA,
    'nucleus':     common.COMMON_SAMPLER_TYPE_TOP_P,
    'typical-p':   common.COMMON_SAMPLER_TYPE_TYPICAL_P,
    'typical':     common.COMMON_SAMPLER_TYPE_TYPICAL_P,
    'min-p':       common.COMMON_SAMPLER_TYPE_MIN_P,
    'temp':        common.COMMON_SAMPLER_TYPE_TEMPERATURE,
    'repetition':  common.COMMON_SAMPLER_TYPE_PENALTIES,
    'penalty':     common.COMMON_SAMPLER_TYPE_PENALTIES,
    'adaptive-p':  common.COMMON_SAMPLER_TYPE_ADAPTIVE_P,
}


def type_to_chr(common.common_sampler_type cnstr) -> str:
    return _SAMPLER_TYPE_TO_CHR.get(<int>cnstr, '?')

def type_to_str(common.common_sampler_type cnstr) -> str:
    return _SAMPLER_TYPE_TO_STR.get(<int>cnstr, '')

def types_from_names(list[str] names, bint allow_alt_names) -> list:
    result = []
    for name in names:
        lower = name.lower()
        if lower in _SAMPLER_NAME_TO_TYPE:
            result.append(_SAMPLER_NAME_TO_TYPE[lower])
        elif allow_alt_names and lower in _SAMPLER_ALT_NAME_TO_TYPE:
            result.append(_SAMPLER_ALT_NAME_TO_TYPE[lower])
    return result

def types_from_chars(str chars) -> list:
    chr_to_type = {v: k for k, v in _SAMPLER_TYPE_TO_CHR.items()}
    return [chr_to_type[c] for c in chars if c in chr_to_type]


# -------------------------------------------------------------------------
# CommonSampler: reimplemented using public llama_sampler API

cdef class CommonSampler:
    """Sampler built from CommonParamsSampling using the public llama_sampler chain API.

    Replaces the previous wrapper around the internal common_sampler C++ struct.
    Grammar is included in the sampler chain (using lazy patterns when triggers
    are configured), which matches the public API's intended usage pattern.
    """
    cdef llama.llama_sampler * chain
    cdef llama.llama_sampler * grmr
    cdef object prev           # deque of previous tokens
    cdef int n_prev
    cdef bint owner

    def __cinit__(self):
        self.chain = NULL
        self.grmr = NULL
        self.owner = True

    def __init__(self, LlamaModel model, CommonParamsSampling params):
        cdef llama.llama_sampler_chain_params lparams = llama.llama_sampler_chain_default_params()
        lparams.no_perf = params.p.no_perf

        self.chain = llama.llama_sampler_chain_init(lparams)
        if self.chain is NULL:
            raise ValueError("Failed to init sampler chain")

        self.n_prev = max(32, params.p.n_prev)
        self.prev = deque(maxlen=self.n_prev)

        cdef const llama.llama_vocab * vocab = llama.llama_model_get_vocab(model.ptr)

        # Grammar sampler (separate from chain, for rejection sampling)
        self.grmr = NULL
        cdef std_string grammar_str = params.p.grammar
        if grammar_str.size() > 0:
            self._init_grammar(vocab, params)

        # Logit bias
        if params.p.logit_bias.size() > 0:
            llama.llama_sampler_chain_add(self.chain,
                llama.llama_sampler_init_logit_bias(
                    llama.llama_vocab_n_tokens(vocab),
                    <int32_t>params.p.logit_bias.size(),
                    params.p.logit_bias.data()))

        # Build sampler chain based on mirostat mode
        cdef bint use_adaptive_p = False
        if params.p.mirostat == 0:
            self._build_standard_chain(model, vocab, params, &use_adaptive_p)
            if use_adaptive_p:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_adaptive_p(
                        params.p.adaptive_target, params.p.adaptive_decay, params.p.seed))
            else:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_dist(params.p.seed))
        elif params.p.mirostat == 1:
            llama.llama_sampler_chain_add(self.chain,
                llama.llama_sampler_init_temp(params.p.temp))
            llama.llama_sampler_chain_add(self.chain,
                llama.llama_sampler_init_mirostat(
                    llama.llama_vocab_n_tokens(vocab),
                    params.p.seed, params.p.mirostat_tau, params.p.mirostat_eta, 100))
        elif params.p.mirostat == 2:
            llama.llama_sampler_chain_add(self.chain,
                llama.llama_sampler_init_temp(params.p.temp))
            llama.llama_sampler_chain_add(self.chain,
                llama.llama_sampler_init_mirostat_v2(
                    params.p.seed, params.p.mirostat_tau, params.p.mirostat_eta))

    cdef void _init_grammar(self, const llama.llama_vocab * vocab, CommonParamsSampling params):
        """Initialize grammar sampler from params."""
        cdef std_vector[const char *] trigger_patterns_c
        cdef std_vector[std_string] trigger_patterns_storage
        cdef std_vector[llama.llama_token] trigger_tokens
        cdef size_t i
        cdef common.common_grammar_trigger trig
        cdef std_string val
        cdef std_string anchored

        for i in range(params.p.grammar_triggers.size()):
            trig = params.p.grammar_triggers[i]
            if trig.type == common.COMMON_GRAMMAR_TRIGGER_TYPE_WORD:
                trigger_patterns_storage.push_back(trig.value)
            elif trig.type == common.COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN:
                trigger_patterns_storage.push_back(trig.value)
            elif trig.type == common.COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL:
                val = trig.value
                anchored = std_string(b"")
                if val.size() > 0:
                    if val.at(0) != <char>ord('^'):
                        anchored = std_string(b"^")
                    anchored += val
                    if val.at(val.size()-1) != <char>ord('$'):
                        anchored += std_string(b"$")
                else:
                    anchored = std_string(b"^$")
                trigger_patterns_storage.push_back(anchored)
            elif trig.type == common.COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN:
                trigger_tokens.push_back(trig.token)

        for i in range(trigger_patterns_storage.size()):
            trigger_patterns_c.push_back(trigger_patterns_storage[i].c_str())

        if params.p.grammar_lazy and (trigger_patterns_c.size() > 0 or trigger_tokens.size() > 0):
            self.grmr = llama.llama_sampler_init_grammar_lazy_patterns(
                vocab,
                params.p.grammar.c_str(),
                b"root",
                trigger_patterns_c.data() if trigger_patterns_c.size() > 0 else NULL,
                trigger_patterns_c.size(),
                trigger_tokens.data() if trigger_tokens.size() > 0 else NULL,
                trigger_tokens.size())
        else:
            self.grmr = llama.llama_sampler_init_grammar(
                vocab, params.p.grammar.c_str(), b"root")

    cdef void _build_standard_chain(self, LlamaModel model,
                                     const llama.llama_vocab * vocab,
                                     CommonParamsSampling params,
                                     bint * use_adaptive_p):
        """Add samplers to chain based on params.samplers order."""
        cdef std_vector[const char *] c_breakers
        cdef common.common_sampler_type stype
        cdef size_t i, j

        for i in range(params.p.samplers.size()):
            stype = params.p.samplers[i]

            if stype == common.COMMON_SAMPLER_TYPE_DRY:
                c_breakers.clear()
                for j in range(params.p.dry_sequence_breakers.size()):
                    c_breakers.push_back(params.p.dry_sequence_breakers[j].c_str())
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_dry(
                        vocab,
                        llama.llama_model_n_ctx_train(model.ptr),
                        params.p.dry_multiplier,
                        params.p.dry_base,
                        params.p.dry_allowed_length,
                        params.p.dry_penalty_last_n,
                        c_breakers.data() if c_breakers.size() > 0 else NULL,
                        c_breakers.size()))

            elif stype == common.COMMON_SAMPLER_TYPE_TOP_K:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_top_k(params.p.top_k))

            elif stype == common.COMMON_SAMPLER_TYPE_TOP_P:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_top_p(params.p.top_p, params.p.min_keep))

            elif stype == common.COMMON_SAMPLER_TYPE_TOP_N_SIGMA:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_top_n_sigma(params.p.top_n_sigma))

            elif stype == common.COMMON_SAMPLER_TYPE_MIN_P:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_min_p(params.p.min_p, params.p.min_keep))

            elif stype == common.COMMON_SAMPLER_TYPE_XTC:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_xtc(
                        params.p.xtc_probability, params.p.xtc_threshold,
                        params.p.min_keep, params.p.seed))

            elif stype == common.COMMON_SAMPLER_TYPE_TYPICAL_P:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_typical(params.p.typ_p, params.p.min_keep))

            elif stype == common.COMMON_SAMPLER_TYPE_TEMPERATURE:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_temp_ext(
                        params.p.temp, params.p.dynatemp_range, params.p.dynatemp_exponent))

            elif stype == common.COMMON_SAMPLER_TYPE_INFILL:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_infill(vocab))

            elif stype == common.COMMON_SAMPLER_TYPE_PENALTIES:
                llama.llama_sampler_chain_add(self.chain,
                    llama.llama_sampler_init_penalties(
                        params.p.penalty_last_n,
                        params.p.penalty_repeat,
                        params.p.penalty_freq,
                        params.p.penalty_present))

            elif stype == common.COMMON_SAMPLER_TYPE_ADAPTIVE_P:
                use_adaptive_p[0] = True

    def __dealloc__(self):
        if self.owner:
            if self.grmr is not NULL:
                llama.llama_sampler_free(self.grmr)
                self.grmr = NULL
            if self.chain is not NULL:
                llama.llama_sampler_free(self.chain)
                self.chain = NULL

    def accept(self, llama.llama_token token, bint accept_grammar):
        """Accept a token. If accept_grammar is true, also accept in grammar sampler."""
        if self.grmr is not NULL and accept_grammar:
            llama.llama_sampler_accept(self.grmr, token)
        llama.llama_sampler_accept(self.chain, token)
        self.prev.append(token)

    def reset(self):
        """Reset the sampler chain and clear token history."""
        llama.llama_sampler_reset(self.chain)
        if self.grmr is not NULL:
            llama.llama_sampler_reset(self.grmr)
        self.prev.clear()

    def clone(self) -> CommonSampler:
        """Clone this sampler."""
        cdef CommonSampler wrapper = CommonSampler.__new__(CommonSampler)
        wrapper.chain = llama.llama_sampler_clone(self.chain)
        wrapper.grmr = llama.llama_sampler_clone(self.grmr) if self.grmr is not NULL else NULL
        wrapper.n_prev = self.n_prev
        wrapper.prev = deque(self.prev, maxlen=self.n_prev)
        wrapper.owner = True
        return wrapper

    def sample(self, LlamaContext ctx, int idx) -> int:
        """Sample a token from the context at the given index.

        If a grammar sampler is configured, uses rejection sampling:
        1. Sample using the chain
        2. Check if the token passes grammar
        3. If not, apply grammar first then resample
        """
        cdef llama.llama_token id
        cdef int n_vocab
        cdef int i
        cdef float * logits
        cdef llama.llama_token_data single_token_data
        cdef llama.llama_token_data_array single_token_data_array
        cdef std_vector[llama.llama_token_data] cur
        cdef llama.llama_token_data_array cur_p

        if self.grmr is NULL:
            # No grammar: just sample from the chain
            return llama.llama_sampler_sample(self.chain, ctx.ptr, idx)

        # With grammar: rejection sampling approach
        # First, sample without grammar constraints (fast path)
        id = llama.llama_sampler_sample(self.chain, ctx.ptr, idx)

        # Check if token passes grammar
        single_token_data.id = id
        single_token_data.logit = 1.0
        single_token_data.p = 0.0
        single_token_data_array.data = &single_token_data
        single_token_data_array.size = 1
        single_token_data_array.selected = -1
        single_token_data_array.sorted = False

        llama.llama_sampler_apply(self.grmr, &single_token_data_array)

        if single_token_data_array.data[0].logit != -INFINITY:
            # Token passes grammar
            return id

        # Token rejected by grammar: resample with grammar applied first
        n_vocab = llama.llama_vocab_n_tokens(
            llama.llama_model_get_vocab(llama.llama_get_model(ctx.ptr)))
        logits = llama.llama_get_logits_ith(ctx.ptr, idx)

        cur.resize(n_vocab)
        for i in range(n_vocab):
            cur[i].id = <llama.llama_token>i
            cur[i].logit = logits[i]
            cur[i].p = 0.0

        cur_p.data = cur.data()
        cur_p.size = cur.size()
        cur_p.selected = -1
        cur_p.sorted = False

        # Apply grammar constraints first, then the sampling chain
        llama.llama_sampler_apply(self.grmr, &cur_p)
        llama.llama_sampler_apply(self.chain, &cur_p)

        if cur_p.selected < 0:
            raise RuntimeError("No token selected during grammar-constrained sampling")

        return cur_p.data[cur_p.selected].id

    def sample_and_accept_n(self, LlamaContext ctx, list[int] draft) -> list[int]:
        """Sample and accept tokens, comparing against draft tokens.

        Cross-references sampled tokens with draft tokens and accepts matches.
        Stops when a mismatch is found.

        Returns at least 1 token, up to len(draft) + 1.
        """
        result = []
        cdef int i
        cdef llama.llama_token id

        for i in range(len(draft)):
            id = self.sample(ctx, i)
            self.accept(id, True)
            result.append(id)
            if draft[i] != id:
                return result

        # All draft tokens matched, sample one more
        id = self.sample(ctx, <int>len(draft))
        self.accept(id, True)
        result.append(id)
        return result

    def get_seed(self) -> int:
        """Get the random seed used by the sampler."""
        return llama.llama_sampler_get_seed(self.chain)

    def get_last(self) -> int:
        """Get the last accepted token."""
        if len(self.prev) == 0:
            raise IndexError("No tokens in sampler history")
        return self.prev[-1]

    def print(self) -> str:
        """Print the sampler chain description."""
        parts = ["logits"]
        cdef int n = llama.llama_sampler_chain_n(self.chain)
        for i in range(n):
            name = llama.llama_sampler_name(
                llama.llama_sampler_chain_get(self.chain, i))
            parts.append("-> " + name.decode() + " ")
        return " ".join(parts)

    def prev_str(self, LlamaContext ctx, int n) -> str:
        """Get a string representation of the last n accepted tokens."""
        cdef int32_t buf_size = 128
        cdef char buf[128]
        cdef int32_t length
        cdef llama.llama_token token_id
        cdef const llama.llama_vocab * vocab = llama.llama_model_get_vocab(
            llama.llama_get_model(ctx.ptr))

        n = min(n, <int>len(self.prev))
        if n <= 0:
            return ""
        result = ""
        prev_list = list(self.prev)
        for i in range(len(prev_list) - n, len(prev_list)):
            token_id = <llama.llama_token>prev_list[i]
            length = llama.llama_token_to_piece(vocab, token_id, buf, buf_size, 0, True)
            if length > 0:
                result += buf[:length].decode('utf-8', errors='replace')
        return result
