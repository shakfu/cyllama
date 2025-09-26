# distutils: language = c++

from libc.stdlib cimport malloc, calloc, realloc, free

cimport whisper as wh

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

    def __init__(self, strategy=wh.WHISPER_SAMPLING_GREEDY):
        self._c_params = wh.whisper_full_default_params(strategy)
        self._language_bytes = None

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

    def __init__(self, model_path, WhisperContextParams params=None):
        if params is None:
            params = WhisperContextParams()

        model_path_bytes = model_path.encode('utf-8')
        self._c_ctx = wh.whisper_init_from_file_with_params(model_path_bytes, params._c_params)

        if self._c_ctx == NULL:
            raise RuntimeError(f"Failed to load whisper model from {model_path}")

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
        cdef int result = wh.whisper_encode(self._c_ctx, offset, n_threads)
        if result != 0:
            raise RuntimeError(f"Encoding failed with error {result}")

    # def full(self, samples, WhisperFullParams params=None):
    #     cdef const float * c_samples = NULL
    #     cdef int n_samples = 0
    #     cdef int result = 0

    #     if params is None:
    #         params = WhisperFullParams()

    #     c_samples = <const float *>(<float[::1]>samples).data
    #     n_samples = len(samples)

    #     result = whisper_full(self._c_ctx, params._c_params, c_samples, n_samples)
    #     if result != 0:
    #         raise RuntimeError(f"Whisper full processing failed with error {result}")

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
