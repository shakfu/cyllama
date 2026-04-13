# distutils: language = c++

from libc.stdlib cimport malloc, calloc, realloc, free

from . cimport whisper as wh

# Constants
class WHISPER:
    SAMPLE_RATE = wh.WHISPER_SAMPLE_RATE
    N_FFT = wh.WHISPER_N_FFT
    HOP_LENGTH = wh.WHISPER_HOP_LENGTH
    CHUNK_SIZE = wh.WHISPER_CHUNK_SIZE

# Enums
class WhisperSamplingStrategy:
    GREEDY = wh.WHISPER_SAMPLING_GREEDY
    BEAM_SEARCH = wh.WHISPER_SAMPLING_BEAM_SEARCH

class WhisperAheadsPreset:
    NONE = wh.WHISPER_AHEADS_NONE
    N_TOP_MOST = wh.WHISPER_AHEADS_N_TOP_MOST
    CUSTOM = wh.WHISPER_AHEADS_CUSTOM
    TINY_EN = wh.WHISPER_AHEADS_TINY_EN
    TINY = wh.WHISPER_AHEADS_TINY
    BASE_EN = wh.WHISPER_AHEADS_BASE_EN
    BASE = wh.WHISPER_AHEADS_BASE
    SMALL_EN = wh.WHISPER_AHEADS_SMALL_EN
    SMALL = wh.WHISPER_AHEADS_SMALL
    MEDIUM_EN = wh.WHISPER_AHEADS_MEDIUM_EN
    MEDIUM = wh.WHISPER_AHEADS_MEDIUM
    LARGE_V1 = wh.WHISPER_AHEADS_LARGE_V1
    LARGE_V2 = wh.WHISPER_AHEADS_LARGE_V2
    LARGE_V3 = wh.WHISPER_AHEADS_LARGE_V3
    LARGE_V3_TURBO = wh.WHISPER_AHEADS_LARGE_V3_TURBO


class WhisperGretype:
    END = wh.WHISPER_GRETYPE_END
    ALT = wh.WHISPER_GRETYPE_ALT
    RULE_REF = wh.WHISPER_GRETYPE_RULE_REF
    CHAR = wh.WHISPER_GRETYPE_CHAR
    CHAR_NOT = wh.WHISPER_GRETYPE_CHAR_NOT
    CHAR_RNG_UPPER = wh.WHISPER_GRETYPE_CHAR_RNG_UPPER
    CHAR_ALT = wh.WHISPER_GRETYPE_CHAR_ALT


# Python wrapper classes
cdef class WhisperContextParams:
    cdef wh.whisper_context_params _c_params

    def __init__(self):
        self._c_params = wh.whisper_context_default_params()

    @property
    def use_gpu(self):
        return self._c_params.use_gpu

    @use_gpu.setter
    def use_gpu(self, bint value):
        self._c_params.use_gpu = value

    @property
    def flash_attn(self):
        return self._c_params.flash_attn

    @flash_attn.setter
    def flash_attn(self, bint value):
        self._c_params.flash_attn = value

    @property
    def gpu_device(self):
        return self._c_params.gpu_device

    @gpu_device.setter
    def gpu_device(self, int value):
        self._c_params.gpu_device = value

    @property
    def dtw_token_timestamps(self):
        return self._c_params.dtw_token_timestamps

    @dtw_token_timestamps.setter
    def dtw_token_timestamps(self, bint value):
        self._c_params.dtw_token_timestamps = value

cdef class WhisperVadParams:
    cdef wh.whisper_vad_params _c_params

    def __init__(self):
        self._c_params = wh.whisper_vad_default_params()

    @property
    def threshold(self):
        return self._c_params.threshold

    @threshold.setter
    def threshold(self, float value):
        self._c_params.threshold = value

    @property
    def min_speech_duration_ms(self):
        return self._c_params.min_speech_duration_ms

    @min_speech_duration_ms.setter
    def min_speech_duration_ms(self, int value):
        self._c_params.min_speech_duration_ms = value

    @property
    def min_silence_duration_ms(self):
        return self._c_params.min_silence_duration_ms

    @min_silence_duration_ms.setter
    def min_silence_duration_ms(self, int value):
        self._c_params.min_silence_duration_ms = value

    @property
    def max_speech_duration_s(self):
        return self._c_params.max_speech_duration_s

    @max_speech_duration_s.setter
    def max_speech_duration_s(self, float value):
        self._c_params.max_speech_duration_s = value

    @property
    def speech_pad_ms(self):
        return self._c_params.speech_pad_ms

    @speech_pad_ms.setter
    def speech_pad_ms(self, int value):
        self._c_params.speech_pad_ms = value

    @property
    def samples_overlap(self):
        return self._c_params.samples_overlap

    @samples_overlap.setter
    def samples_overlap(self, float value):
        self._c_params.samples_overlap = value


cdef class WhisperFullParams:
    cdef wh.whisper_full_params _c_params
    cdef bytes _language_bytes  # Keep bytes object alive for language parameter
    cdef bytes _initial_prompt_bytes
    cdef bytes _suppress_regex_bytes
    cdef bytes _vad_model_path_bytes

    def __init__(self, strategy=wh.WHISPER_SAMPLING_GREEDY):
        self._c_params = wh.whisper_full_default_params(strategy)
        self._language_bytes = None
        self._initial_prompt_bytes = None
        self._suppress_regex_bytes = None
        self._vad_model_path_bytes = None

    @property
    def strategy(self):
        return self._c_params.strategy

    @strategy.setter
    def strategy(self, wh.whisper_sampling_strategy value):
        self._c_params.strategy = value

    @property
    def n_threads(self):
        return self._c_params.n_threads

    @n_threads.setter
    def n_threads(self, int value):
        self._c_params.n_threads = value

    @property
    def n_max_text_ctx(self):
        return self._c_params.n_max_text_ctx

    @n_max_text_ctx.setter
    def n_max_text_ctx(self, int value):
        self._c_params.n_max_text_ctx = value

    @property
    def offset_ms(self):
        return self._c_params.offset_ms

    @offset_ms.setter
    def offset_ms(self, int value):
        self._c_params.offset_ms = value

    @property
    def duration_ms(self):
        return self._c_params.duration_ms

    @duration_ms.setter
    def duration_ms(self, int value):
        self._c_params.duration_ms = value

    @property
    def translate(self):
        return self._c_params.translate

    @translate.setter
    def translate(self, bint value):
        self._c_params.translate = value

    @property
    def no_context(self):
        return self._c_params.no_context

    @no_context.setter
    def no_context(self, bint value):
        self._c_params.no_context = value

    @property
    def no_timestamps(self):
        return self._c_params.no_timestamps

    @no_timestamps.setter
    def no_timestamps(self, bint value):
        self._c_params.no_timestamps = value

    @property
    def single_segment(self):
        return self._c_params.single_segment

    @single_segment.setter
    def single_segment(self, bint value):
        self._c_params.single_segment = value

    @property
    def print_special(self):
        return self._c_params.print_special

    @print_special.setter
    def print_special(self, bint value):
        self._c_params.print_special = value

    @property
    def print_progress(self):
        return self._c_params.print_progress

    @print_progress.setter
    def print_progress(self, bint value):
        self._c_params.print_progress = value

    @property
    def print_realtime(self):
        return self._c_params.print_realtime

    @print_realtime.setter
    def print_realtime(self, bint value):
        self._c_params.print_realtime = value

    @property
    def print_timestamps(self):
        return self._c_params.print_timestamps

    @print_timestamps.setter
    def print_timestamps(self, bint value):
        self._c_params.print_timestamps = value

    @property
    def token_timestamps(self):
        return self._c_params.token_timestamps

    @token_timestamps.setter
    def token_timestamps(self, bint value):
        self._c_params.token_timestamps = value

    @property
    def temperature(self):
        return self._c_params.temperature

    @temperature.setter
    def temperature(self, float value):
        self._c_params.temperature = value

    @property
    def language(self):
        if self._c_params.language == NULL:
            return None
        return self._c_params.language.decode('utf-8')

    @language.setter
    def language(self, value):
        if value is None:
            self._c_params.language = NULL
            self._language_bytes = None
        else:
            self._language_bytes = value.encode('utf-8')
            self._c_params.language = <const char*>self._language_bytes

    # -----------------------------------------------------------------------
    # Token-level timestamp parameters
    # -----------------------------------------------------------------------

    @property
    def thold_pt(self):
        """timestamp token probability threshold"""
        return self._c_params.thold_pt

    @thold_pt.setter
    def thold_pt(self, float value):
        self._c_params.thold_pt = value

    @property
    def thold_ptsum(self):
        """timestamp token sum probability threshold"""
        return self._c_params.thold_ptsum

    @thold_ptsum.setter
    def thold_ptsum(self, float value):
        self._c_params.thold_ptsum = value

    @property
    def max_len(self):
        """max segment length in characters"""
        return self._c_params.max_len

    @max_len.setter
    def max_len(self, int value):
        self._c_params.max_len = value

    @property
    def split_on_word(self):
        """split on word rather than on token"""
        return self._c_params.split_on_word

    @split_on_word.setter
    def split_on_word(self, bint value):
        self._c_params.split_on_word = value

    @property
    def max_tokens(self):
        """max tokens per segment (0 = no limit)"""
        return self._c_params.max_tokens

    @max_tokens.setter
    def max_tokens(self, int value):
        self._c_params.max_tokens = value

    # -----------------------------------------------------------------------
    # Speed-up / debug
    # -----------------------------------------------------------------------

    @property
    def debug_mode(self):
        """enable debug mode (eg. dump log_mel)"""
        return self._c_params.debug_mode

    @debug_mode.setter
    def debug_mode(self, bint value):
        self._c_params.debug_mode = value

    @property
    def audio_ctx(self):
        """overwrite the audio context size (0 = use default)"""
        return self._c_params.audio_ctx

    @audio_ctx.setter
    def audio_ctx(self, int value):
        self._c_params.audio_ctx = value

    @property
    def tdrz_enable(self):
        """enable tinydiarize speaker turn detection"""
        return self._c_params.tdrz_enable

    @tdrz_enable.setter
    def tdrz_enable(self, bint value):
        self._c_params.tdrz_enable = value

    # -----------------------------------------------------------------------
    # Prompt / regex
    # -----------------------------------------------------------------------

    @property
    def suppress_regex(self):
        """regex to suppress tokens matching this pattern"""
        if self._c_params.suppress_regex == NULL:
            return None
        return self._c_params.suppress_regex.decode('utf-8')

    @suppress_regex.setter
    def suppress_regex(self, value):
        if value is None:
            self._c_params.suppress_regex = NULL
            self._suppress_regex_bytes = None
        else:
            self._suppress_regex_bytes = value.encode('utf-8')
            self._c_params.suppress_regex = <const char*>self._suppress_regex_bytes

    @property
    def initial_prompt(self):
        """initial prompt to condition the model (prepended to every decode window)"""
        if self._c_params.initial_prompt == NULL:
            return None
        return self._c_params.initial_prompt.decode('utf-8')

    @initial_prompt.setter
    def initial_prompt(self, value):
        if value is None:
            self._c_params.initial_prompt = NULL
            self._initial_prompt_bytes = None
        else:
            self._initial_prompt_bytes = value.encode('utf-8')
            self._c_params.initial_prompt = <const char*>self._initial_prompt_bytes

    @property
    def carry_initial_prompt(self):
        """if true, always prepend initial_prompt to every decode window"""
        return self._c_params.carry_initial_prompt

    @carry_initial_prompt.setter
    def carry_initial_prompt(self, bint value):
        self._c_params.carry_initial_prompt = value

    # -----------------------------------------------------------------------
    # Language detection
    # -----------------------------------------------------------------------

    @property
    def detect_language(self):
        """if true, auto-detect spoken language"""
        return self._c_params.detect_language

    @detect_language.setter
    def detect_language(self, bint value):
        self._c_params.detect_language = value

    # -----------------------------------------------------------------------
    # Decoding / suppression
    # -----------------------------------------------------------------------

    @property
    def suppress_blank(self):
        """suppress blank outputs at the beginning of sampling"""
        return self._c_params.suppress_blank

    @suppress_blank.setter
    def suppress_blank(self, bint value):
        self._c_params.suppress_blank = value

    @property
    def suppress_nst(self):
        """suppress non-speech tokens"""
        return self._c_params.suppress_nst

    @suppress_nst.setter
    def suppress_nst(self, bint value):
        self._c_params.suppress_nst = value

    @property
    def max_initial_ts(self):
        """max initial timestamp (1.0 = no limit)"""
        return self._c_params.max_initial_ts

    @max_initial_ts.setter
    def max_initial_ts(self, float value):
        self._c_params.max_initial_ts = value

    @property
    def length_penalty(self):
        """length penalty (-1.0 = default from model)"""
        return self._c_params.length_penalty

    @length_penalty.setter
    def length_penalty(self, float value):
        self._c_params.length_penalty = value

    # -----------------------------------------------------------------------
    # Temperature fallback
    # -----------------------------------------------------------------------

    @property
    def temperature_inc(self):
        """temperature increment on fallback (0.0 = disabled)"""
        return self._c_params.temperature_inc

    @temperature_inc.setter
    def temperature_inc(self, float value):
        self._c_params.temperature_inc = value

    @property
    def entropy_thold(self):
        """entropy threshold for decoder fail (triggers temperature fallback)"""
        return self._c_params.entropy_thold

    @entropy_thold.setter
    def entropy_thold(self, float value):
        self._c_params.entropy_thold = value

    @property
    def logprob_thold(self):
        """avg logprob threshold for decoder fail (triggers temperature fallback)"""
        return self._c_params.logprob_thold

    @logprob_thold.setter
    def logprob_thold(self, float value):
        self._c_params.logprob_thold = value

    @property
    def no_speech_thold(self):
        """no speech probability threshold"""
        return self._c_params.no_speech_thold

    @no_speech_thold.setter
    def no_speech_thold(self, float value):
        self._c_params.no_speech_thold = value

    # -----------------------------------------------------------------------
    # Greedy / beam search strategy params
    # -----------------------------------------------------------------------

    @property
    def greedy_best_of(self):
        """number of best candidates to keep for greedy strategy"""
        return wh.whisper_params_get_greedy_best_of(&self._c_params)

    @greedy_best_of.setter
    def greedy_best_of(self, int value):
        wh.whisper_params_set_greedy_best_of(&self._c_params, value)

    @property
    def beam_size(self):
        """beam size for beam search strategy"""
        return wh.whisper_params_get_beam_size(&self._c_params)

    @beam_size.setter
    def beam_size(self, int value):
        wh.whisper_params_set_beam_size(&self._c_params, value)

    @property
    def beam_patience(self):
        """beam search patience factor"""
        return wh.whisper_params_get_beam_patience(&self._c_params)

    @beam_patience.setter
    def beam_patience(self, float value):
        wh.whisper_params_set_beam_patience(&self._c_params, value)

    # -----------------------------------------------------------------------
    # Grammar
    # -----------------------------------------------------------------------

    @property
    def grammar_penalty(self):
        """penalty applied to grammar-violating tokens"""
        return self._c_params.grammar_penalty

    @grammar_penalty.setter
    def grammar_penalty(self, float value):
        self._c_params.grammar_penalty = value

    # -----------------------------------------------------------------------
    # VAD (Voice Activity Detection)
    # -----------------------------------------------------------------------

    @property
    def vad(self):
        """enable voice activity detection"""
        return self._c_params.vad

    @vad.setter
    def vad(self, bint value):
        self._c_params.vad = value

    @property
    def vad_model_path(self):
        """path to VAD model file"""
        if self._c_params.vad_model_path == NULL:
            return None
        return self._c_params.vad_model_path.decode('utf-8')

    @vad_model_path.setter
    def vad_model_path(self, value):
        if value is None:
            self._c_params.vad_model_path = NULL
            self._vad_model_path_bytes = None
        else:
            self._vad_model_path_bytes = value.encode('utf-8')
            self._c_params.vad_model_path = <const char*>self._vad_model_path_bytes


cdef class WhisperTokenData:
    cdef wh.whisper_token_data _c_data

    def __init__(self):
        pass

    @property
    def id(self):
        return self._c_data.id

    @property
    def tid(self):
        return self._c_data.tid

    @property
    def p(self):
        return self._c_data.p

    @property
    def plog(self):
        return self._c_data.plog

    @property
    def pt(self):
        return self._c_data.pt

    @property
    def ptsum(self):
        return self._c_data.ptsum

    @property
    def t0(self):
        return self._c_data.t0

    @property
    def t1(self):
        return self._c_data.t1

    @property
    def t_dtw(self):
        return self._c_data.t_dtw

    @property
    def vlen(self):
        return self._c_data.vlen


cdef class WhisperContext:
    cdef wh.whisper_context * _c_ctx
    # `readonly` so external code (notably the concurrency-guard
    # regression tests) can call `_busy_lock.acquire(blocking=False)` /
    # `release()` to simulate "another thread is in flight" without
    # needing to actually run a slow whisper_full pass on a worker
    # thread. The lock is still set internally by __init__; readonly
    # only blocks Python-level rebinding.
    cdef readonly object _busy_lock

    def __init__(self, model_path, WhisperContextParams params=None):
        import threading
        from cyllama._validation import validate_whisper_file

        if params is None:
            params = WhisperContextParams()

        # Whisper accepts both legacy ggml and newer GGUF magics. Validate
        # that the file exists, is readable, non-empty, AND has a known
        # whisper magic in its first 4 bytes -- otherwise whisper.cpp will
        # try to parse arbitrary garbage and may segfault before returning.
        validate_whisper_file(model_path, kind="whisper model")

        model_path_bytes = model_path.encode('utf-8')
        self._c_ctx = wh.whisper_init_from_file_with_params(model_path_bytes, params._c_params)

        if self._c_ctx == NULL:
            raise RuntimeError(
                f"Failed to load whisper model from {model_path}. "
                "The file passed basic checks but whisper.cpp could not load it. "
                "Possible causes: unsupported model format/version, corrupt file, "
                "or insufficient memory."
            )

        # Concurrent-use guard for the underlying whisper.cpp context.
        # whisper_context is not thread-safe and full() / encode()
        # release the GIL during native calls, so two threads racing on
        # the same context corrupt internal state. We use a non-blocking
        # lock acquired around each guarded method: legitimate sequential
        # ownership transfer (asyncio.to_thread, ThreadPoolExecutor) keeps
        # working, but real concurrent use raises a clear RuntimeError.
        # __dealloc__ is intentionally NOT guarded because gc may run it
        # on any thread.
        self._busy_lock = threading.Lock()

    def _try_acquire_busy(self):
        """Acquire the busy-lock or raise on contention."""
        if not self._busy_lock.acquire(blocking=False):
            raise RuntimeError(
                "WhisperContext is currently being used by another thread. "
                "whisper.cpp contexts are not thread-safe -- create one "
                "WhisperContext per thread instead of sharing a single "
                "instance across threads."
            )

    def __dealloc__(self):
        if self._c_ctx != NULL:
            wh.whisper_free(self._c_ctx)
            self._c_ctx = NULL

    def version(self):
        return wh.whisper_version().decode('utf-8')

    def system_info(self):
        return wh.whisper_print_system_info().decode('utf-8')

    def n_vocab(self):
        return wh.whisper_n_vocab(self._c_ctx)

    def n_text_ctx(self):
        return wh.whisper_n_text_ctx(self._c_ctx)

    def n_audio_ctx(self):
        return wh.whisper_n_audio_ctx(self._c_ctx)

    def is_multilingual(self):
        return bool(wh.whisper_is_multilingual(self._c_ctx))

    def model_n_vocab(self):
        return wh.whisper_model_n_vocab(self._c_ctx)

    def model_n_audio_ctx(self):
        return wh.whisper_model_n_audio_ctx(self._c_ctx)

    def model_n_audio_state(self):
        return wh.whisper_model_n_audio_state(self._c_ctx)

    def model_n_audio_head(self):
        return wh.whisper_model_n_audio_head(self._c_ctx)

    def model_n_audio_layer(self):
        return wh.whisper_model_n_audio_layer(self._c_ctx)

    def model_n_text_ctx(self):
        return wh.whisper_model_n_text_ctx(self._c_ctx)

    def model_n_text_state(self):
        return wh.whisper_model_n_text_state(self._c_ctx)

    def model_n_text_head(self):
        return wh.whisper_model_n_text_head(self._c_ctx)

    def model_n_text_layer(self):
        return wh.whisper_model_n_text_layer(self._c_ctx)

    def model_n_mels(self):
        return wh.whisper_model_n_mels(self._c_ctx)

    def model_ftype(self):
        return wh.whisper_model_ftype(self._c_ctx)

    def model_type(self):
        return wh.whisper_model_type(self._c_ctx)

    def model_type_readable(self):
        return wh.whisper_model_type_readable(self._c_ctx).decode('utf-8')

    def token_to_str(self, int token):
        cdef const char * result = wh.whisper_token_to_str(self._c_ctx, token)
        if result == NULL:
            return ""
        return result.decode('utf-8')

    def token_eot(self):
        return wh.whisper_token_eot(self._c_ctx)

    def token_sot(self):
        return wh.whisper_token_sot(self._c_ctx)

    def token_solm(self):
        return wh.whisper_token_solm(self._c_ctx)

    def token_prev(self):
        return wh.whisper_token_prev(self._c_ctx)

    def token_nosp(self):
        return wh.whisper_token_nosp(self._c_ctx)

    def token_not(self):
        return wh.whisper_token_not(self._c_ctx)

    def token_beg(self):
        return wh.whisper_token_beg(self._c_ctx)

    def token_lang(self, int lang_id):
        return wh.whisper_token_lang(self._c_ctx, lang_id)

    def token_translate(self):
        return wh.whisper_token_translate(self._c_ctx)

    def token_transcribe(self):
        return wh.whisper_token_transcribe(self._c_ctx)

    def tokenize(self, str text, int max_tokens=512):
        cdef int n_tokens = 0
        cdef bytes text_bytes = text.encode('utf-8')
        cdef wh.whisper_token * tokens = <wh.whisper_token *>malloc(max_tokens * sizeof(wh.whisper_token))
        if tokens is NULL:
            raise MemoryError("Failed to allocate token buffer")

        try:
            n_tokens = wh.whisper_tokenize(self._c_ctx, text_bytes, tokens, max_tokens)
            if n_tokens < 0:
                raise RuntimeError(f"Tokenization failed, need {-n_tokens} tokens but only {max_tokens} provided")

            result = []
            for i in range(n_tokens):
                result.append(tokens[i])
            return result
        finally:
            free(tokens)

    def token_count(self, str text):
        text_bytes = text.encode('utf-8')
        return wh.whisper_token_count(self._c_ctx, text_bytes)

    def lang_max_id(self):
        return wh.whisper_lang_max_id()

    def lang_id(self, str lang):
        lang_bytes = lang.encode('utf-8')
        return wh.whisper_lang_id(lang_bytes)

    def lang_str(self, int id):
        cdef const char * result = wh.whisper_lang_str(id)
        if result == NULL:
            return None
        return result.decode('utf-8')

    def lang_str_full(self, int id):
        cdef const char * result = wh.whisper_lang_str_full(id)
        if result == NULL:
            return None
        return result.decode('utf-8')

    # def pcm_to_mel(self, samples, int n_threads=1):
    #     cdef const float * c_samples = <const float *>(<float[::1]>samples).data
    #     cdef int n_samples = len(samples)

    #     cdef int result = whisper_pcm_to_mel(self._c_ctx, c_samples, n_samples, n_threads)
    #     if result != 0:
    #         raise RuntimeError(f"PCM to mel conversion failed with error {result}")

    def encode(self, int offset=0, int n_threads=1):
        self._try_acquire_busy()
        cdef int result = 0
        try:
            result = wh.whisper_encode(self._c_ctx, offset, n_threads)
        finally:
            self._busy_lock.release()
        if result != 0:
            raise RuntimeError(f"Encoding failed with error {result}")

    def full(self, samples, WhisperFullParams params=None):
        cdef const float * c_samples = NULL
        cdef int n_samples = 0
        cdef int result = 0
        cdef float[::1] samples_view
        cdef wh.whisper_context * ctx
        cdef wh.whisper_full_params c_params

        if params is None:
            params = WhisperFullParams()

        samples_view = samples
        c_samples = &samples_view[0]
        n_samples = len(samples)

        ctx = self._c_ctx
        c_params = params._c_params

        self._try_acquire_busy()
        try:
            with nogil:
                result = wh.whisper_full(ctx, c_params, c_samples, n_samples)
        finally:
            self._busy_lock.release()
        if result != 0:
            raise RuntimeError(f"Whisper full processing failed with error {result}")
        return result

    def full_n_segments(self):
        return wh.whisper_full_n_segments(self._c_ctx)

    def full_lang_id(self):
        return wh.whisper_full_lang_id(self._c_ctx)

    def full_get_segment_t0(self, int i_segment):
        return wh.whisper_full_get_segment_t0(self._c_ctx, i_segment)

    def full_get_segment_t1(self, int i_segment):
        return wh.whisper_full_get_segment_t1(self._c_ctx, i_segment)

    def full_get_segment_text(self, int i_segment):
        cdef const char * result = wh.whisper_full_get_segment_text(self._c_ctx, i_segment)
        if result == NULL:
            return ""
        return result.decode('utf-8')

    def full_n_tokens(self, int i_segment):
        return wh.whisper_full_n_tokens(self._c_ctx, i_segment)

    def full_get_token_text(self, int i_segment, int i_token):
        cdef const char * result = wh.whisper_full_get_token_text(self._c_ctx, i_segment, i_token)
        if result == NULL:
            return ""
        return result.decode('utf-8')

    def full_get_token_id(self, int i_segment, int i_token):
        return wh.whisper_full_get_token_id(self._c_ctx, i_segment, i_token)

    def full_get_token_data(self, int i_segment, int i_token):
        cdef wh.whisper_token_data c_data = wh.whisper_full_get_token_data(self._c_ctx, i_segment, i_token)

        data = WhisperTokenData()
        data._c_data = c_data
        return data

    def full_get_token_p(self, int i_segment, int i_token):
        return wh.whisper_full_get_token_p(self._c_ctx, i_segment, i_token)

    def full_get_segment_no_speech_prob(self, int i_segment):
        return wh.whisper_full_get_segment_no_speech_prob(self._c_ctx, i_segment)

    def print_timings(self):
        wh.whisper_print_timings(self._c_ctx)

    def reset_timings(self):
        wh.whisper_reset_timings(self._c_ctx)

cdef class WhisperState:
    cdef wh.whisper_state * _c_state
    cdef WhisperContext _ctx

    def __init__(self, WhisperContext ctx):
        self._ctx = ctx
        self._c_state = wh.whisper_init_state(ctx._c_ctx)

        if self._c_state == NULL:
            raise RuntimeError("Failed to initialize whisper state")

    def __dealloc__(self):
        if self._c_state != NULL:
            wh.whisper_free_state(self._c_state)
            self._c_state = NULL

# Module-level functions
def ggml_backend_load_all():
    """Load all available ggml backends (CUDA, Metal, Vulkan, etc.).

    Must be called before creating a WhisperContext so that GPU backends
    are registered and available for inference.
    """
    import os
    from .._backend_dl import libs_to_load
    _dir = os.path.dirname(os.path.abspath(__file__))
    # In dynamic builds the backend libs live alongside the llama extension
    _llama_dir = os.path.join(os.path.dirname(_dir), "llama")
    if os.path.isdir(_llama_dir):
        wh.ggml_backend_load_all_from_path(_llama_dir.encode())
    else:
        wh.ggml_backend_load_all_from_path(_dir.encode())
    # Wheel repair tools (auditwheel/delvewheel) place backend libs in a
    # cyllama_<variant>.libs/ directory and rename them with content hashes.
    # ggml's built-in discovery breaks on renamed files, so we load each
    # candidate individually — ggml_backend_load() silently skips files
    # that are not valid backends.
    _site = os.path.dirname(os.path.dirname(_dir))  # site-packages/
    for _path in libs_to_load(_site):
        wh.ggml_backend_load(_path)

cdef void _no_log_cb(wh.ggml_log_level level, const char * text, void * user_data) noexcept:
    pass

def disable_logging():
    """Suppress all C-level log output from whisper.cpp and ggml."""
    wh.whisper_log_set(_no_log_cb, NULL)

def version():
    return wh.whisper_version().decode('utf-8')

def print_system_info():
    return wh.whisper_print_system_info().decode('utf-8')

def lang_max_id():
    return wh.whisper_lang_max_id()

def lang_id(str lang):
    lang_bytes = lang.encode('utf-8')
    return wh.whisper_lang_id(lang_bytes)

def lang_str(int id):
    cdef const char * result = wh.whisper_lang_str(id)
    if result == NULL:
        return None
    return result.decode('utf-8')

def lang_str_full(int id):
    cdef const char * result = wh.whisper_lang_str_full(id)
    if result == NULL:
        return None
    return result.decode('utf-8')
