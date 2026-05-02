# distutils: language = c++

from libc.stdlib cimport malloc, calloc, realloc, free
from libcpp cimport bool as cppbool  # required for func pointer sigs

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


# =============================================================================
# Streaming-callback trampolines for WhisperFullParams.
#
# whisper.cpp invokes these from inside whisper_full() while the GIL is
# released. Each trampoline:
#   1. reacquires the GIL (`with gil`)
#   2. retrieves the WhisperFullParams instance from user_data
#   3. invokes the stored Python callable
#   4. catches any user-callback exception (raising across the C boundary
#      is undefined behaviour even with `noexcept with gil`) and prints
#      the traceback so failures surface for debugging
#
# The Python callable is stored on the params instance, and the params
# instance itself is the user_data pointer. Lifetime invariant: the
# params object must outlive the whisper_full() call -- which is the
# normal case, since `WhisperContext.full(samples, params)` holds a
# reference to `params` in its local frame for the duration of the call.
# =============================================================================

cdef void _whisper_new_segment_trampoline(
    wh.whisper_context * ctx, wh.whisper_state * state, int n_new, void * user_data
) noexcept with gil:
    cdef WhisperFullParams params = <WhisperFullParams>user_data
    cb = params._new_segment_callback
    if cb is None:
        return
    try:
        cb(n_new)
    except BaseException:
        import traceback
        traceback.print_exc()


cdef void _whisper_progress_trampoline(
    wh.whisper_context * ctx, wh.whisper_state * state, int progress, void * user_data
) noexcept with gil:
    cdef WhisperFullParams params = <WhisperFullParams>user_data
    cb = params._progress_callback
    if cb is None:
        return
    try:
        cb(progress)
    except BaseException:
        import traceback
        traceback.print_exc()


cdef cppbool _whisper_encoder_begin_trampoline(
    wh.whisper_context * ctx, wh.whisper_state * state, void * user_data
) noexcept with gil:
    cdef WhisperFullParams params = <WhisperFullParams>user_data
    cb = params._encoder_begin_callback
    if cb is None:
        # No callback set: proceed with encoding (the C-side default).
        return True
    try:
        result = cb()
        # Treat any non-False return as "proceed". This matches the
        # idiomatic Python use of bool() on user-returned values.
        return bool(result)
    except BaseException:
        import traceback
        traceback.print_exc()
        # Fail-safe: proceed with encoding rather than aborting on a
        # buggy filter -- consistent with progress_callback in the llama
        # side, which returns True on exception so model loading
        # continues.
        return True


cdef class WhisperFullParams:
    cdef wh.whisper_full_params _c_params
    cdef bytes _language_bytes  # Keep bytes object alive for language parameter
    cdef bytes _initial_prompt_bytes
    cdef bytes _suppress_regex_bytes
    cdef bytes _vad_model_path_bytes
    # Python callables for streaming callbacks. None when no callback
    # is registered. Held on the params instance so they outlive the
    # whisper_full() call as long as the params instance does.
    cdef object _new_segment_callback
    cdef object _progress_callback
    cdef object _encoder_begin_callback

    def __init__(self, strategy=wh.WHISPER_SAMPLING_GREEDY):
        self._c_params = wh.whisper_full_default_params(strategy)
        self._language_bytes = None
        self._initial_prompt_bytes = None
        self._suppress_regex_bytes = None
        self._vad_model_path_bytes = None
        self._new_segment_callback = None
        self._progress_callback = None
        self._encoder_begin_callback = None

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

    # ------------------------------------------------------------------
    # Streaming callbacks
    #
    # Register a Python callable to be invoked from inside whisper_full().
    # Pass None to clear. Each callback runs while the GIL has been
    # reacquired by the trampoline, so the callable can do any normal
    # Python work -- but it executes in the middle of a transcription,
    # so keep the body short. A long callback blocks the decoder.
    #
    # The params instance itself is the user_data pointer passed to
    # whisper.cpp, so it must outlive the full() call. That's already
    # the natural case (callers hold a reference for the duration of
    # `ctx.full(samples, params)`), but if you stash params in a place
    # that gets GC'd mid-call, the callback fires on a freed object.
    # ------------------------------------------------------------------

    def set_new_segment_callback(self, callback):
        """Register a callback fired after each new segment is produced.

        Args:
            callback: ``Callable[[int], None]`` where the int is the
                number of *new* segments since the last invocation.
                Use it as an offset into :meth:`WhisperContext.full_n_segments`
                to read just the new segments::

                    def on_new_segments(n_new):
                        n_total = ctx.full_n_segments()
                        for i in range(n_total - n_new, n_total):
                            print(ctx.full_get_segment_text(i))

                    params.set_new_segment_callback(on_new_segments)
                    ctx.full(samples, params)

                Pass ``None`` to clear a previously-registered callback.

        Exceptions raised by the callback are caught and traceback-printed
        but otherwise swallowed -- raising into the C decoder is undefined
        behaviour. The callback runs while the busy-lock is held, so
        result accessors (``full_n_segments``, ``full_get_segment_text``,
        etc.) work but a re-entrant ``full()`` call would deadlock.
        """
        self._new_segment_callback = callback
        if callback is None:
            self._c_params.new_segment_callback = NULL
            self._c_params.new_segment_callback_user_data = NULL
        else:
            self._c_params.new_segment_callback = _whisper_new_segment_trampoline
            self._c_params.new_segment_callback_user_data = <void*>self

    def set_progress_callback(self, callback):
        """Register a callback fired periodically with progress percent.

        Args:
            callback: ``Callable[[int], None]`` where the int is in
                ``[0, 100]``. Pass ``None`` to clear.

        Same exception-handling and reentrancy notes as
        :meth:`set_new_segment_callback`.
        """
        self._progress_callback = callback
        if callback is None:
            self._c_params.progress_callback = NULL
            self._c_params.progress_callback_user_data = NULL
        else:
            self._c_params.progress_callback = _whisper_progress_trampoline
            self._c_params.progress_callback_user_data = <void*>self

    def set_encoder_begin_callback(self, callback):
        """Register a callback fired before encoder runs; return False to abort.

        Args:
            callback: ``Callable[[], bool]`` returning True to proceed
                with encoding or False to abort the current chunk.
                Pass ``None`` to clear.

        Useful for cooperative cancellation (e.g. checking a stop flag
        between chunks). Exceptions raised by the callback are caught,
        traceback-printed, and treated as ``True`` (proceed) -- aborting
        on a buggy filter would be more destructive than continuing.
        """
        self._encoder_begin_callback = callback
        if callback is None:
            self._c_params.encoder_begin_callback = NULL
            self._c_params.encoder_begin_callback_user_data = NULL
        else:
            self._c_params.encoder_begin_callback = _whisper_encoder_begin_trampoline
            self._c_params.encoder_begin_callback_user_data = <void*>self


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
    """Whisper transcription context.

    Loads a whisper.cpp model file and exposes the inference + accessor
    surface as Python methods. Method names mirror the underlying C API
    (``whisper_full`` -> :meth:`full`, ``whisper_full_get_segment_text``
    -> :meth:`full_get_segment_text`, etc.) so upstream documentation at
    https://github.com/ggml-org/whisper.cpp maps directly. The ``full_``
    prefix on accessor methods reflects that they read state populated by
    a prior :meth:`full` call -- it is part of the C symbol name, not a
    separate concept.

    Typical end-to-end usage::

        import numpy as np
        from cyllama.whisper.whisper_cpp import (
            WhisperContext, WhisperContextParams, WhisperFullParams,
        )

        with WhisperContext("ggml-base.en.bin", WhisperContextParams()) as ctx:
            samples = load_pcm_16khz_mono_float32(...)  # 1-D float32 ndarray
            ctx.full(samples, WhisperFullParams())
            for i in range(ctx.full_n_segments()):
                t0 = ctx.full_get_segment_t0(i)  # 10 ms units
                t1 = ctx.full_get_segment_t1(i)
                text = ctx.full_get_segment_text(i)
                print(f"[{t0/100:.2f} -> {t1/100:.2f}] {text}")

    Thread safety: a single ``WhisperContext`` is **not** thread-safe;
    :meth:`full` and :meth:`encode` release the GIL, so two threads
    racing on the same instance corrupt internal state. The non-blocking
    ``_busy_lock`` raises ``RuntimeError`` on contention rather than
    serializing or corrupting -- create one context per worker thread.
    """
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
        from cyllama.utils.validation import validate_whisper_file

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

    def close(self):
        """Release the underlying whisper context immediately. Idempotent."""
        if self._c_ctx != NULL:
            wh.whisper_free(self._c_ctx)
            self._c_ctx = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def version(self):
        """whisper.cpp library version string (e.g. ``"1.7.0"``)."""
        return wh.whisper_version().decode('utf-8')

    def system_info(self):
        """Human-readable build/CPU/backend feature summary, multi-line.

        Includes whether AVX/AVX2/AVX512/NEON/Metal/CUDA are compiled in
        and which are active at runtime. Useful for bug reports.
        """
        return wh.whisper_print_system_info().decode('utf-8')

    def n_vocab(self):
        """Number of tokens in the model's vocabulary."""
        return wh.whisper_n_vocab(self._c_ctx)

    def n_text_ctx(self):
        """Maximum text-decoder context length in tokens."""
        return wh.whisper_n_text_ctx(self._c_ctx)

    def n_audio_ctx(self):
        """Audio-encoder context length in mel frames."""
        return wh.whisper_n_audio_ctx(self._c_ctx)

    def is_multilingual(self):
        """True if the loaded model supports non-English input.

        ``.en`` checkpoints (e.g. ``ggml-base.en.bin``) return False;
        the equivalent multilingual checkpoints return True.
        """
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
        """Render a single token id back to its text fragment.

        Returns ``""`` for tokens with no string form (e.g. special tokens
        like ``<|endoftext|>`` if the model returns NULL for them).
        """
        cdef const char * result = wh.whisper_token_to_str(self._c_ctx, token)
        if result == NULL:
            return ""
        return result.decode('utf-8')

    def token_eot(self):
        """End-of-transcript special token id (``<|endoftext|>``)."""
        return wh.whisper_token_eot(self._c_ctx)

    def token_sot(self):
        """Start-of-transcript special token id (``<|startoftranscript|>``)."""
        return wh.whisper_token_sot(self._c_ctx)

    def token_solm(self):
        """Start-of-language-model special token id."""
        return wh.whisper_token_solm(self._c_ctx)

    def token_prev(self):
        """``<|prev|>`` special token id (used to feed prior context)."""
        return wh.whisper_token_prev(self._c_ctx)

    def token_nosp(self):
        """``<|nospeech|>`` special token id."""
        return wh.whisper_token_nosp(self._c_ctx)

    def token_not(self):
        """``<|notimestamps|>`` special token id."""
        return wh.whisper_token_not(self._c_ctx)

    def token_beg(self):
        """``<|0.00|>`` segment-begin special token id (timestamp anchor)."""
        return wh.whisper_token_beg(self._c_ctx)

    def token_lang(self, int lang_id):
        """Token id for the language tag corresponding to ``lang_id``.

        ``lang_id`` is the integer id from :meth:`lang_id`.
        """
        return wh.whisper_token_lang(self._c_ctx, lang_id)

    def token_translate(self):
        """``<|translate|>`` task-token id."""
        return wh.whisper_token_translate(self._c_ctx)

    def token_transcribe(self):
        """``<|transcribe|>`` task-token id."""
        return wh.whisper_token_transcribe(self._c_ctx)

    def tokenize(self, str text, int max_tokens=512):
        """Tokenize ``text`` to a list of whisper token ids.

        Args:
            text: Input string.
            max_tokens: Buffer capacity. If the tokenizer needs more
                tokens than this, the call raises ``RuntimeError`` with
                the required size in the message; retry with a larger
                ``max_tokens``.

        Raises:
            MemoryError: token buffer allocation failed.
            RuntimeError: ``max_tokens`` was too small (message reports
                the required count).
        """
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
        """Count the tokens ``text`` would produce, without allocating."""
        text_bytes = text.encode('utf-8')
        return wh.whisper_token_count(self._c_ctx, text_bytes)

    def lang_max_id(self):
        """Largest valid language id (use as upper bound when iterating)."""
        return wh.whisper_lang_max_id()

    def lang_id(self, str lang):
        """Convert an ISO 639-1 code (e.g. ``"en"``, ``"es"``) to its int id.

        Returns -1 if ``lang`` is not a recognised whisper language code.
        """
        lang_bytes = lang.encode('utf-8')
        return wh.whisper_lang_id(lang_bytes)

    def lang_str(self, int id):
        """Convert a language id back to its ISO 639-1 code (e.g. ``"en"``).

        Returns ``None`` if ``id`` is out of range.
        """
        cdef const char * result = wh.whisper_lang_str(id)
        if result == NULL:
            return None
        return result.decode('utf-8')

    def lang_str_full(self, int id):
        """Convert a language id to its English name (e.g. ``"english"``).

        Returns ``None`` if ``id`` is out of range.
        """
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
        """Run the audio encoder pass on already-mel'd input.

        Low-level building block used by :meth:`full` internally.
        Most callers should use :meth:`full` instead -- it runs the
        complete log-mel + encoder + decoder + segment-extraction
        pipeline in one call.

        Args:
            offset: Mel-frame offset to start encoding from.
            n_threads: CPU threads for the encoder pass.

        Raises:
            RuntimeError: Encoding failed (non-zero return from
                ``whisper_encode``).
            RuntimeError: Another thread is currently using this context
                (``WhisperContext`` is not thread-safe).
        """
        self._try_acquire_busy()
        cdef int result = 0
        try:
            result = wh.whisper_encode(self._c_ctx, offset, n_threads)
        finally:
            self._busy_lock.release()
        if result != 0:
            raise RuntimeError(f"Encoding failed with error {result}")

    def full(self, samples, WhisperFullParams params=None):
        """Run whisper transcription on PCM audio samples.

        Args:
            samples: 1-D C-contiguous float32 buffer of mono PCM samples
                at 16 kHz (typically a numpy array, but any object
                supporting the buffer protocol with the right layout
                works). Length must be > 0.
            params: Decoding parameters; uses defaults if None.

        Raises:
            TypeError: samples is not a numpy float32 array (when numpy
                is the input type) or fails the buffer-protocol bind.
            ValueError: samples is not 1-D, not C-contiguous, or empty.
        """
        cdef const float * c_samples = NULL
        cdef int n_samples = 0
        cdef int result = 0
        cdef float[::1] samples_view
        cdef wh.whisper_context * ctx
        cdef wh.whisper_full_params c_params

        if params is None:
            params = WhisperFullParams()

        # Validate up front so the error message names the actual problem.
        # Without this check, mistyped numpy inputs raise a generic
        # "Buffer dtype mismatch" from the memoryview cast below, and a
        # Python list raises "a bytes-like object is required" -- neither
        # of which makes the audio-format requirement obvious.
        try:
            import numpy as _np
        except ImportError:
            _np = None
        if _np is not None and isinstance(samples, _np.ndarray):
            if samples.dtype != _np.float32:
                raise TypeError(
                    f"samples must be a float32 array, got dtype={samples.dtype}. "
                    "Cast with samples.astype(np.float32) before calling."
                )
            if samples.ndim != 1:
                raise ValueError(
                    f"samples must be 1-D mono PCM, got {samples.ndim}-D shape "
                    f"{samples.shape}. For stereo input, mix down to mono first."
                )
            if not samples.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "samples must be C-contiguous; "
                    "use np.ascontiguousarray(samples) to fix."
                )
        if len(samples) == 0:
            raise ValueError("samples is empty; pass at least one PCM sample")

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

    # ------------------------------------------------------------------
    # Result accessors. Read state populated by the most recent full()
    # call. Calling these before full() is undefined-behavior territory
    # (whisper.cpp returns whatever the uninitialized state contains).
    # ------------------------------------------------------------------

    def full_n_segments(self):
        """Number of segments in the most recent transcription.

        A segment is a contiguous span of recognized speech with its own
        start/end timestamps. Use as the upper bound for iterating with
        :meth:`full_get_segment_text`, :meth:`full_get_segment_t0`, etc.
        """
        return wh.whisper_full_n_segments(self._c_ctx)

    def full_lang_id(self):
        """Detected language id of the most recent transcription.

        Combine with :meth:`lang_str` for the ISO code (``"en"``) or
        :meth:`lang_str_full` for the English name (``"english"``).
        Returns -1 if the model didn't run language detection
        (e.g. ``.en`` checkpoint, or ``language`` was forced in params).
        """
        return wh.whisper_full_lang_id(self._c_ctx)

    def full_get_segment_t0(self, int i_segment):
        """Start time of segment ``i_segment`` in **10 ms units**.

        Divide by 100 to get seconds. This is whisper.cpp's native
        timebase -- not milliseconds. ``i_segment`` must be in
        ``[0, full_n_segments())``.
        """
        return wh.whisper_full_get_segment_t0(self._c_ctx, i_segment)

    def full_get_segment_t1(self, int i_segment):
        """End time of segment ``i_segment`` in **10 ms units**.

        Divide by 100 to get seconds. See :meth:`full_get_segment_t0`.
        """
        return wh.whisper_full_get_segment_t1(self._c_ctx, i_segment)

    def full_get_segment_text(self, int i_segment):
        """Transcribed text of segment ``i_segment`` as a UTF-8 string.

        Returns ``""`` if the C side returned NULL (out-of-range index
        or empty segment).
        """
        cdef const char * result = wh.whisper_full_get_segment_text(self._c_ctx, i_segment)
        if result == NULL:
            return ""
        return result.decode('utf-8')

    def full_n_tokens(self, int i_segment):
        """Number of tokens in segment ``i_segment``.

        Use as the upper bound for iterating with
        :meth:`full_get_token_text` / :meth:`full_get_token_id` / etc.
        """
        return wh.whisper_full_n_tokens(self._c_ctx, i_segment)

    def full_get_token_text(self, int i_segment, int i_token):
        """Text fragment for token ``i_token`` of segment ``i_segment``.

        Returns ``""`` if the C side returned NULL.
        """
        cdef const char * result = wh.whisper_full_get_token_text(self._c_ctx, i_segment, i_token)
        if result == NULL:
            return ""
        return result.decode('utf-8')

    def full_get_token_id(self, int i_segment, int i_token):
        """Vocabulary id for token ``i_token`` of segment ``i_segment``."""
        return wh.whisper_full_get_token_id(self._c_ctx, i_segment, i_token)

    def full_get_token_data(self, int i_segment, int i_token):
        """Full :class:`WhisperTokenData` for one token.

        Includes the token id, log-probability, sampled probability,
        timestamp probability, and (when DTW token-timestamps are
        enabled in :class:`WhisperContextParams`) the DTW-aligned start
        and end times.
        """
        cdef wh.whisper_token_data c_data = wh.whisper_full_get_token_data(self._c_ctx, i_segment, i_token)

        data = WhisperTokenData()
        data._c_data = c_data
        return data

    def full_get_token_p(self, int i_segment, int i_token):
        """Sampled probability of token ``i_token`` in ``[0.0, 1.0]``.

        Convenience accessor; same as ``full_get_token_data(...).p``.
        """
        return wh.whisper_full_get_token_p(self._c_ctx, i_segment, i_token)

    def full_get_segment_no_speech_prob(self, int i_segment):
        """No-speech probability for segment ``i_segment`` in ``[0.0, 1.0]``.

        High values indicate the segment is likely silence/noise rather
        than spoken content. Usable as a confidence filter to drop
        spurious segments from the output.
        """
        return wh.whisper_full_get_segment_no_speech_prob(self._c_ctx, i_segment)

    def print_timings(self):
        """Print accumulated timing breakdown to stderr.

        Reports load / mel / sample / encode / decode / batchd / prompt
        wall-clock totals. Useful for profiling. Timings accumulate
        across calls; use :meth:`reset_timings` to zero them.
        """
        wh.whisper_print_timings(self._c_ctx)

    def reset_timings(self):
        """Zero out the timing counters reported by :meth:`print_timings`."""
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

    def close(self):
        """Release the underlying whisper state immediately. Idempotent."""
        if self._c_state != NULL:
            wh.whisper_free_state(self._c_state)
            self._c_state = NULL

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

# Module-level functions
def ggml_backend_load_all():
    """Load all available ggml backends (CUDA, Metal, Vulkan, etc.).

    Must be called before creating a WhisperContext so that GPU backends
    are registered and available for inference.
    """
    import os
    from .._internal.backend_dl import libs_to_load
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
