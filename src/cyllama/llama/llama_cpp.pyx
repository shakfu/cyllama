# distutils: language = c++
"""cyllama: a thin cython wrapper of llama.cpp"""


from libc.stdint cimport uint8_t, int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport strlen
from libc.stdlib cimport malloc, calloc, realloc, free
from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string
from libcpp cimport bool as cppbool  # required for func pointer sigs

cimport ggml
cimport llama
cimport common
cimport sampling
cimport chat
cimport log


import os
# from enum import Enum
from typing import Optional, Sequence, Callable

# exports
# -----------------------------------------------------------------------------

# __all__ = [
#     'GgmlBackendDevice',
#     'GgmlBackend',
#     'GgmlTensor',
#     'GgmlThreadPoolParams',
#     'GgmlThreadPool',
#     
#     'LlamaTokenData',
#     'LlamaTokenDataArray',
#     'LlamaBatch',
#     'LlamaModelKvOverride',
#     'LlamaModelTensorBuftOverride',
#     'LlamaModelParams',
#     'LlamaContextParams',
#     'LlamaModelQuantizeParams',
#     'LlamaLogitBias',
#     'LlamaSamplerChainParams',
#     'LlamaChatMessage',
#     'LlamaVocab',
#     'LlamaModel',
#     'LlamaContext',
#     'LlamaSampler',
#     'LlamaAdapterLora',
# ]

# includes
# -----------------------------------------------------------------------------

include "common.pxi"
include "sampling.pxi"
include "tts_helpers.pxi"


# constants
# -----------------------------------------------------------------------------

cpdef enum:
    GGML_DEFAULT_N_THREADS = 4
    GGML_MAX_DIMS = 4
    GGML_MAX_N_THREADS = 16
    GGML_MAX_NAME = 64
    GGML_MAX_OP_PARAMS = 64
    GGML_MAX_SRC = 10

cpdef enum:
    GGML_ROPE_TYPE_NEOX = 2
    GGML_ROPE_TYPE_MROPE = 8
    GGML_ROPE_TYPE_VISION = 24


# callbacks
# -----------------------------------------------------------------------------

cdef void log_callback(ggml.ggml_log_level level, const char * text, void * py_log_callback) noexcept:
    """ggml_log_callback wrapper to enabling python callbacks to be used"""
    (<object>py_log_callback)(level, text.decode())


def set_log_callback(object py_log_callback):
    """Set callback for all future logging events.

    If this is not called, or NULL is supplied, everything is output on stderr.
    """
    llama.llama_log_set(<ggml.ggml_log_callback>&log_callback, <void*>py_log_callback)


cdef bint abort_callback(void * py_abort_callback) noexcept:
    """ggml_abort_callback wrapper enabling python callbacks to be used"""
    return (<object>py_abort_callback)()


cdef cppbool sched_eval_callback(ggml.ggml_tensor * t, cppbool ask, void * py_sched_eval_callback) noexcept:
    """ggml_backend_sched_eval_callback wrapper enabling python callbacks to be used"""
    cdef GgmlTensor tensor = GgmlTensor.from_ptr(t)
    return (<object>py_sched_eval_callback)(tensor, ask)


cdef cppbool progress_callback(float progress, void * py_progress_callback) noexcept:
    """llama_progress_callback callback wrapper enabling python callbacks to be used"""
    return (<object>py_progress_callback)(progress)


# high-level api
# -----------------------------------------------------------------------------


# def ask(str prompt, str model, n_predict=512, n_ctx=2048, disable_log=True, n_threads=4) -> str:
#     """ask/prompt a llama model"""

#     cdef str result = llama.simple_prompt(
#         model.encode(),
#         prompt.encode(),
#         n_predict,
#         n_ctx,
#         disable_log,
#         n_threads).decode()
#     return result.strip()


# ggml wrapper classes
# -----------------------------------------------------------------------------

cdef class GgmlBackendDevice:
    cdef ggml.ggml_backend_dev_t ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            free(self.ptr)
            self.ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef GgmlBackendDevice from_ptr(ggml.ggml_backend_dev_t ptr, bint owner=False):
        cdef GgmlBackendDevice wrapper = GgmlBackendDevice.__new__(GgmlBackendDevice)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper


cdef class GgmlBackend:
    cdef ggml.ggml_backend_t ptr
    cdef bint owner

    def __cinit__(self, GgmlBackendDevice dev, str params):
        self.ptr = ggml.ggml_backend_dev_init(dev.ptr, params.encode())
        self.owner = True

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef GgmlBackendDevice from_ptr(ggml.ggml_backend_dev_t ptr, bint owner=False):
        cdef GgmlBackendDevice wrapper = GgmlBackendDevice.__new__(GgmlBackendDevice)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper


cdef class GgmlTensor:
    cdef ggml.ggml_tensor * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            free(self.ptr)
            self.ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef GgmlTensor from_ptr(ggml.ggml_tensor *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef GgmlTensor wrapper = GgmlTensor.__new__(GgmlTensor)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @staticmethod
    cdef GgmlTensor create():
        cdef ggml.ggml_tensor *ptr = <ggml.ggml_tensor *>malloc(sizeof(ggml.ggml_tensor))
        if ptr is NULL:
            raise MemoryError
        # ptr.a = 0
        # ptr.b = 0
        return GgmlTensor.from_ptr(ptr, owner=True)


cdef class GgmlThreadPoolParams:
    # NOTE: should this be a * ptr
    cdef ggml.ggml_threadpool_params p


    def __init__(self, int n_threads):
        self.p = ggml.ggml_threadpool_params_default(n_threads)

    # cdef void ggml_threadpool_params_init(ggml_threadpool_params * p, int n_threads)

    @staticmethod
    cdef GgmlThreadPoolParams from_instance(ggml.ggml_threadpool_params params):
        cdef GgmlThreadPoolParams wrapper = GgmlThreadPoolParams.__new__(GgmlThreadPoolParams)
        wrapper.p = params
        return wrapper

    # @staticmethod
    # def from_cpu_params(CpuParams params) -> GgmlThreadPoolParams:
    #     cdef ggml.ggml_threadpool_params tparams = ggml.ggml_threadpool_params_from_cpu_params(params.ptr[0])
    #     return GgmlThreadPoolParams.from_instance(tparams)

    def match(self, GgmlThreadPoolParams other) -> bool:
        return ggml.ggml_threadpool_params_match(&self.p, &other.p)

    @property
    def cpumask(self) -> list[bool]:
        """mask of cpu cores (all-zeros means use default affinity settings)

        cpumask[GGML_MAX_N_THREADS] is (by default) of size 16
        """
        res = []
        for i in range(GGML_MAX_N_THREADS):
            res.append(<bint>self.p.cpumask[i])
        return res

    @cpumask.setter
    def cpumask(self, values: list[bool]):
        assert len(values) == GGML_MAX_N_THREADS
        for i in range(GGML_MAX_N_THREADS):
            self.p.cpumask[i] = <bint>values[i]

    @property
    def n_threads(self) -> int:
        """number of threads"""
        return self.p.n_threads

    @n_threads.setter
    def n_threads(self, int value):
        self.p.n_threads = value

    @property
    def prio(self) -> ggml.ggml_sched_priority:
        """thread priority"""
        return self.p.prio

    @prio.setter
    def prio(self, ggml.ggml_sched_priority value):
        self.p.prio = value

    @property
    def poll(self) -> uint32_t:
        """polling level (0 - no polling, 100 - aggressive polling)"""
        return self.p.poll

    @poll.setter
    def poll(self, uint32_t value):
        self.p.poll = value

    @property
    def strict_cpu(self) -> bool:
        """strict cpu placement"""
        return self.p.strict_cpu

    @strict_cpu.setter
    def strict_cpu(self, bint value):
        self.p.strict_cpu = value

    @property
    def paused(self) -> bool:
        """start in paused state"""
        return self.p.paused

    @paused.setter
    def paused(self, bint value):
        self.p.paused = value


cdef class GgmlThreadPool:
    cdef ggml.ggml_threadpool * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, GgmlThreadPoolParams params):
        self.ptr = ggml.ggml_threadpool_new(&params.p)
        if self.ptr is NULL:
            raise MemoryError
        self.owner = True

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            ggml.ggml_threadpool_free(self.ptr)
            self.ptr = NULL

    # def get_n_threads(self) -> int:
    #     return ggml.ggml_threadpool_get_n_threads(self.ptr)

    def pause(self):
        return ggml.ggml_threadpool_pause(self.ptr)

    def resume(self):
        return ggml.ggml_threadpool_resume(self.ptr)


# llama wrapper classes
# -----------------------------------------------------------------------------

cdef class LlamaTokenData:
    cdef public llama.llama_token id
    cdef public float logit
    cdef public float p

    def __cinit__(self, llama.llama_token id, float logit, float p):
        self.id = id
        self.logit = logit
        self.p = p


cdef class LlamaTokenDataArray:
    cdef list[LlamaTokenData] data
    cdef size_t size
    cdef int64_t selected
    cdef bint is_sorted


cdef class LlamaBatch:
    """Input data for llama_decode

    A llama_batch object can contain input about one or many sequences
    The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens

    - token  : the token ids of the input (used when embd is NULL)
    - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    - pos    : the positions of the respective token in the sequence
               (if set to NULL, the token position will be tracked automatically by llama_decode)
    - seq_id : the sequence to which the respective token belongs
               (if set to NULL, the sequence ID will be assumed to be 0)
    - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
               (if set to NULL, only the logits for last token will be returned)
    """
    cdef llama.llama_batch p
    cdef int _n_tokens
    cdef public int embd
    cdef public int n_seq_max
    cdef public bint verbose
    cdef bint owner

    def __init__(self, *, n_tokens: int, embd: int, n_seq_max: int, verbose: bool = True):
        """Allocates a batch of tokens on the heap that can hold a maximum of `n_tokens`

        Each token can be assigned up to `n_seq_max sequence` ids

        If `embd != 0`, `llama_batch.embd` will be allocated with size of `n_tokens * embd * sizeof(float)`
        Otherwise, `llama_batch.token` will be allocated to store `n_tokens llama_token`
        The rest of the llama_batch members are allocated with size `n_tokens`
        """
        self._n_tokens = n_tokens
        self.embd = embd
        self.n_seq_max = n_seq_max
        self.verbose = verbose
        self.owner = True

        self.p = llama.llama_batch_init(
            self._n_tokens, self.embd, self.n_seq_max
        )

    def __dealloc__(self):
        """The batch has to be freed with `llama_batch_free()`"""
        if self.owner is True:
            llama.llama_batch_free(self.p)

    def close(self):
        self.__dealloc__()

    @staticmethod
    cdef LlamaBatch from_instance(llama.llama_batch batch):
        cdef LlamaBatch wrapper = LlamaBatch.__new__(LlamaBatch)
        wrapper.p = batch
        wrapper._n_tokens = batch.n_tokens
        wrapper.embd = 0 if batch.embd == NULL else 1
        wrapper.n_seq_max = 1  # Default for llama_batch_get_one
        wrapper.verbose = True
        wrapper.owner = False  # Important: we don't own this batch, so don't free it
        return wrapper

    @property
    def n_tokens(self) -> int:
        return self.p.n_tokens

    def reset(self):
        self.p.n_tokens = 0

    # def add(self, llama.llama_token id, llama.llama_pos pos, list[int] seq_ids, bint logits):
    #     cdef std_vector[llama.llama_seq_id] _seq_ids
    #     for i in seq_ids:
    #         _seq_ids.push_back(i)
    #     llama.common_batch_add(self.p, id, pos, _seq_ids, logits)

    # def clear(self):
    #     llama.common_batch_clear(self.p)

    def set_batch(self, batch: Sequence[int], n_past: int, logits_all: bool):
        n_tokens = len(batch)
        self.p.n_tokens = n_tokens
        for i in range(n_tokens):
            self.p.token[i] = batch[i]
            self.p.pos[i] = n_past + i
            self.p.seq_id[i][0] = 0
            self.p.n_seq_id[i] = 1
            self.p.logits[i] = logits_all
        self.p.logits[n_tokens - 1] = True

    def add_sequence(self, batch: Sequence[int], seq_id: int, logits_all: bool):
        n_tokens = len(batch)
        n_tokens0 = self.p.n_tokens
        self.p.n_tokens += n_tokens
        for i in range(n_tokens):
            j = n_tokens0 + i
            self.p.token[j] = batch[i]
            self.p.pos[j] = i
            self.p.seq_id[j][0] = seq_id
            self.p.n_seq_id[j] = 1
            self.p.logits[j] = logits_all
        self.p.logits[n_tokens - 1] = True

    def set_last_logits_to_true(self):
        self.p.logits[self.p.n_tokens - 1] = True


cdef class LlamaModelKvOverride:
    cdef llama.llama_model_kv_override * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            free(self.ptr)
            self.ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef LlamaModelKvOverride from_ptr(llama.llama_model_kv_override *ptr, bint owner=False):
        cdef LlamaModelKvOverride wrapper = LlamaModelKvOverride.__new__(LlamaModelKvOverride)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @property
    def tag(self) -> llama.llama_model_kv_override_type:
        return self.ptr.tag

    @property
    def key(self) -> str:
        return self.ptr.key.decode()

    @property
    def val_i64(self) -> int:
        return self.ptr.val_i64

    @property
    def val_f64(self) -> float:
        return self.ptr.val_f64

    @property
    def val_bool(self) -> bool:
        return <bint>self.ptr.val_bool

    @property
    def val_str(self) -> str:
        return self.ptr.val_str.decode()


cdef class LlamaModelTensorBuftOverride:
    cdef llama.llama_model_tensor_buft_override * ptr

    @property
    def pattern(self) -> str:
        return self.ptr.pattern.decode()

    # @property
    # def buft(self) -> ggml.ggml_backend_buffer_type_t:
    #     return self.ptr.buft




cdef class LlamaModelParams:
    cdef llama.llama_model_params p

    def __init__(self):
        self.p = llama.llama_model_default_params()
        # self.p.progress_callback = &progress_callback # FIXME: causes crash

    @staticmethod
    cdef LlamaModelParams from_instance(llama.llama_model_params params):
        cdef LlamaModelParams wrapper = LlamaModelParams.__new__(LlamaModelParams)
        wrapper.p = params
        return wrapper

    @property
    def devices(self) -> list[GgmlBackendDevice]:
        """list of devices to use for offloading"""

    @devices.setter
    def devices(self, devs: list[GgmlBackendDevice]):
        """NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)"""

    @property
    def tensor_buft_overrides(self) -> list[LlamaModelTensorBuftOverride]:
        """NULL-terminated list of buffer types to use for tensors that match a pattern"""

    @property
    def n_gpu_layers(self) -> int:
        """Number of layers to store in VRAM."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def split_mode(self) -> llama.llama_split_mode:
        """How to split the model across multiple GPUs."""
        return self.p.split_mode

    @split_mode.setter
    def split_mode(self, llama.llama_split_mode value):
        self.p.split_mode = value

    @property
    def main_gpu(self) -> int:
        """The GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE"""
        return self.p.main_gpu

    @main_gpu.setter
    def main_gpu(self, value: int):
        self.p.main_gpu = value

    @property
    def tensor_split(self) -> list[float]:
        """Proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()"""
        cdef size_t length = llama.llama_max_devices()
        results = []
        if self.p.tensor_split:
            for i in range(length):
                n = <float>self.p.tensor_split[i]
                results.append(n)
        return results

    @property
    def progress_callback(self) -> Callable[[float], bool]:
        """Called with a progress value between 0.0 and 1.0. Pass NULL to disable.

        If the provided progress_callback returns true, model loading continues.
        If it returns false, model loading is immediately aborted.

        progress_callback_user_data is context pointer passed to the progress callback
        """
        return <object>self.p.progress_callback_user_data

    @progress_callback.setter
    def progress_callback(self, object py_progress_callback):
        self.p.progress_callback_user_data = <void*>py_progress_callback

    @property
    def kv_overrides(self) -> list[LlamaModelKvOverride]:
        """override key-value pairs of the model meta data

        const llama_model_kv_override * kv_overrides
        """




    @property
    def vocab_only(self) -> bool:
        """Load only the vocabulary, no weights"""
        return self.p.vocab_only

    @vocab_only.setter
    def vocab_only(self, value: bool):
        self.p.vocab_only = value

    @property
    def use_mmap(self) -> bool:
        """Use mmap if possible"""
        return self.p.use_mmap

    @use_mmap.setter
    def use_mmap(self, value: bool):
        self.p.use_mmap = value

    @property
    def use_mlock(self) -> bool:
        """Force system to keep model in RAM"""
        return self.p.use_mlock

    @use_mlock.setter
    def use_mlock(self, value: bool):
        self.p.use_mlock = value

    @property
    def check_tensors(self) -> bool:
        """Validate model tensor data"""
        return self.p.check_tensors

    @check_tensors.setter
    def check_tensors(self, value: bool):
        self.p.check_tensors = value


cdef class LlamaContextParams:
    cdef llama.llama_context_params p

    def __init__(self):
        self.p = llama.llama_context_default_params()

    @staticmethod
    cdef LlamaContextParams from_common_params(CommonParams params):
        cdef LlamaContextParams wrapper = LlamaContextParams.__new__(LlamaContextParams)
        wrapper.p = common.common_context_params_to_llama(params.p)
        return wrapper

    @property
    def n_ctx(self) -> int:
        """text context, 0 = from model."""
        return self.p.n_ctx

    @n_ctx.setter
    def n_ctx(self, value: int):
        self.p.n_ctx = value

    @property
    def n_batch(self) -> int:
        """logical maximum batch size that can be submitted to llama_decode."""
        return self.p.n_batch

    @n_batch.setter
    def n_batch(self, value: int):
        self.p.n_batch = value

    @property
    def n_ubatch(self) -> int:
        """physical maximum batch size."""
        return self.p.n_ubatch

    @n_ubatch.setter
    def n_ubatch(self, value: int):
        self.p.n_ubatch = value

    @property
    def n_seq_max(self) -> int:
        """max number of sequences (i.e. distinct states for recurrent models)."""
        return self.p.n_seq_max

    @n_seq_max.setter
    def n_seq_max(self, value: int):
        self.p.n_seq_max = value

    @property
    def n_threads(self) -> int:
        """number of threads to use for generation."""
        return self.p.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self.p.n_threads = value

    @property
    def n_threads_batch(self) -> int:
        """number of threads to use for batch processing"""
        return self.p.n_threads_batch

    @n_threads_batch.setter
    def n_threads_batch(self, value: int):
        self.p.n_threads_batch = value

    @property
    def rope_scaling_type(self) -> llama.llama_rope_scaling_type:
        """number of threads to use for batch processing"""
        return <llama.llama_rope_scaling_type>self.p.rope_scaling_type

    @rope_scaling_type.setter
    def rope_scaling_type(self, llama.llama_rope_scaling_type value):
        self.p.rope_scaling_type = value

    @property
    def pooling_type(self) -> llama.llama_pooling_type:
        """whether to pool (sum) embedding results by sequence id"""
        return <llama.llama_pooling_type>self.p.pooling_type

    @pooling_type.setter
    def pooling_type(self, llama.llama_pooling_type value):
        self.p.pooling_type = value

    @property
    def attention_type(self) -> llama.llama_attention_type:
        """attention type to use for embeddings"""
        return <llama.llama_attention_type>self.p.attention_type

    @attention_type.setter
    def attention_type(self, llama.llama_attention_type value):
        self.p.attention_type = value

    @property
    def rope_freq_base(self) -> float:
        """RoPE base frequency, 0 = from model"""
        return self.p.rope_freq_base

    @rope_freq_base.setter
    def rope_freq_base(self, float value):
        self.p.rope_freq_base = value

    @property
    def rope_freq_scale(self) -> float:
        """RoPE frequency scaling factor."""
        return self.p.rope_freq_scale

    @rope_freq_scale.setter
    def rope_freq_scale(self, value: float):
        self.p.rope_freq_scale = value

    @property
    def yarn_ext_factor(self) -> float:
        """YaRN extrapolation mix factor."""
        return self.p.yarn_ext_factor

    @yarn_ext_factor.setter
    def yarn_ext_factor(self, value: float):
        self.p.yarn_ext_factor = value

    @property
    def yarn_attn_factor(self) -> float:
        """YaRN magnitude scaling factor."""
        return self.p.yarn_attn_factor

    @yarn_attn_factor.setter
    def yarn_attn_factor(self, value: float):
        self.p.yarn_attn_factor = value

    @property
    def yarn_beta_fast(self) -> float:
        """YaRN low correction dim."""
        return self.p.yarn_beta_fast

    @yarn_beta_fast.setter
    def yarn_beta_fast(self, value: float):
        self.p.yarn_beta_fast = value

    @property
    def yarn_beta_slow(self) -> float:
        """YaRN high correction dim."""
        return self.p.yarn_beta_slow

    @yarn_beta_slow.setter
    def yarn_beta_slow(self, value: float):
        self.p.yarn_beta_slow = value

    @property
    def yarn_orig_ctx(self) -> int:
        """YaRN original context length."""
        return self.p.yarn_orig_ctx

    @yarn_orig_ctx.setter
    def yarn_orig_ctx(self, value: int):
        self.p.yarn_orig_ctx = value

    # ggml.ggml_backend_sched_eval_callback cb_eval;

    # @property
    # def cb_eval(self) -> py_sched_eval_callback:
    #     """get/set python ggml backend sched eval callback."""
    #     return <object>self.p.cb_eval_user_data

    # @cb_eval.setter
    # def cb_eval(self, object py_sched_eval_callback):
    #     self.p.cb_eval_user_data = <void*>py_sched_eval_callback

    @property
    def type_k(self) -> ggml.ggml_type:
        """data type for K cache"""
        return <ggml.ggml_type>self.p.type_k

    @type_k.setter
    def type_k(self, ggml.ggml_type value):
        self.p.type_k = value

    @property
    def type_v(self) -> ggml.ggml_type:
        """data type for V cache"""
        return <ggml.ggml_type>self.p.type_v

    @type_v.setter
    def type_v(self, ggml.ggml_type value):
        self.p.type_v = value

    # @property
    # def logits_all(self) -> bool:
    #     """the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)"""
    #     return self.p.logits_all

    # @logits_all.setter
    # def logits_all(self, value: bool):
    #     self.p.logits_all = value

    @property
    def offload_kqv(self) -> bool:
        """whether to offload the KQV ops (including the KV cache) to GPU"""
        return self.p.offload_kqv

    @offload_kqv.setter
    def offload_kqv(self, value: bool):
        self.p.offload_kqv = value

    # @property
    # def flash_attn(self) -> bool:
    #     """whether to use flash attention [EXPERIMENTAL]"""
    #     return self.p.flash_attn

    # @flash_attn.setter
    # def flash_attn(self, value: bool):
    #     self.p.flash_attn = value

    @property
    def no_perf(self) -> bool:
        """whether to measure performance timings"""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, value: bool):
        self.p.no_perf = value

    # ggml.ggml_abort_callback abort_callback;
    # void *              abort_callback_data;


cdef class LlamaModelQuantizeParams:
    cdef llama.llama_model_quantize_params p

    def __init__(self):
        self.p = llama.llama_model_quantize_default_params()

    @staticmethod
    cdef LlamaModelQuantizeParams from_instance(llama.llama_model_quantize_params params):
        cdef LlamaModelQuantizeParams wrapper = LlamaModelQuantizeParams.__new__(LlamaModelQuantizeParams)
        wrapper.p = params
        return wrapper

    @property
    def nthread(self) -> int:
        """number of threads to use for quantizing.

        if <=0 will use std::thread::hardware_concurrency().
        """
        return self.p.nthread

    @nthread.setter
    def nthread(self, value: int):
        self.p.nthread = value

    @property
    def ftype(self) -> llama.llama_ftype:
        """quantize to this llama_ftype"""
        return self.p.ftype

    @ftype.setter
    def ftype(self, value: int):
        self.p.ftype = value

    @property
    def output_tensor_type(self) -> int:
        """output tensor type"""
        return self.p.output_tensor_type

    @output_tensor_type.setter
    def output_tensor_type(self, value: int):
        self.p.output_tensor_type = value

    @property
    def token_embedding_type(self) -> int:
        """itoken embeddings tensor type"""
        return self.p.token_embedding_type

    @token_embedding_type.setter
    def token_embedding_type(self, value: int):
        self.p.token_embedding_type = value

    @property
    def allow_requantize(self) -> bool:
        """allow quantizing non-f32/f16 tensors"""
        return self.p.allow_requantize

    @allow_requantize.setter
    def allow_requantize(self, value: bool):
        self.p.allow_requantize = value

    @property
    def quantize_output_tensor(self) -> bool:
        """quantize output.weight"""
        return self.p.quantize_output_tensor

    @quantize_output_tensor.setter
    def quantize_output_tensor(self, value: bool):
        self.p.quantize_output_tensor = value

    @property
    def only_copy(self) -> bool:
        """only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored"""
        return self.p.only_copy

    @only_copy.setter
    def only_copy(self, value: bool):
        self.p.only_copy = value

    @property
    def pure(self) -> bool:
        """quantize all tensors to the default type"""
        return self.p.pure

    @pure.setter
    def pure(self, value: bool):
        self.p.pure = value

    @property
    def keep_split(self) -> bool:
        """quantize to the same number of shards"""
        return self.p.keep_split

    @keep_split.setter
    def keep_split(self, value: bool):
        self.p.keep_split = value

    @property
    def imatrix(self) -> None:
        """pointer to importance matrix data"""

    @property
    def kv_overrides(self) -> None:
        """pointer to vector containing overrides"""

    @property
    def tensor_types(self) -> None:
        """pointer to vector containing tensor types"""


cdef class LlamaLogitBias:
    cdef llama.llama_logit_bias p

    @property
    def token(self) -> int:
        """token token"""
        return self.p.token

    @token.setter
    def token(self, int value):
        self.p.token = value

    @property
    def bias(self) -> float:
        """bias"""
        return self.p.bias

    @bias.setter
    def bias(self, float value):
        self.p.bias = value


cdef class LlamaSamplerChainParams:
    cdef llama.llama_sampler_chain_params p

    def __init__(self):
        self.p = llama.llama_sampler_chain_default_params()

    @staticmethod
    cdef LlamaSamplerChainParams from_instance(llama.llama_sampler_chain_params params):
        cdef LlamaSamplerChainParams wrapper = LlamaSamplerChainParams.__new__(LlamaSamplerChainParams)
        wrapper.p = params
        return wrapper

    @property
    def no_perf(self) -> bool:
        """whether to measure performance timings."""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, value: bool):
        self.p.no_perf = value


cdef class LlamaChatMessage:
    """cython wrapper for llama.llama_chat_message

    members role and content are const char *
    """
    cdef llama.llama_chat_message p
    cdef bytes _role_bytes
    cdef bytes _content_bytes

    def __init__(self, str role, str content):
        self._role_bytes = role.encode('utf-8')
        self._content_bytes = content.encode('utf-8')
        self.p.role = <const char *>self._role_bytes
        self.p.content = <const char *>self._content_bytes

    @staticmethod
    cdef LlamaChatMessage from_instance(llama.llama_chat_message msg):
        cdef LlamaChatMessage wrapper = LlamaChatMessage.__new__(LlamaChatMessage)
        wrapper.p = msg
        # Store copies of the strings to ensure proper lifetime management
        if msg.role is not NULL:
            wrapper._role_bytes = msg.role
        else:
            wrapper._role_bytes = b""
        if msg.content is not NULL:
            wrapper._content_bytes = msg.content  
        else:
            wrapper._content_bytes = b""
        return wrapper

    cdef llama.llama_chat_message copy(self):
        return self.p

    @property
    def role(self) -> str:
        """chat role"""
        return self.p.role.decode()

    @property
    def content(self) -> str:
        """chat content"""
        return self.p.content.decode()


cdef class LlamaVocab:
    """cython wrapper for llama.llama_vocab"""
    cdef llama.llama_vocab * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    @staticmethod
    cdef LlamaVocab from_ptr(llama.llama_vocab *ptr, bint owner=False):
        cdef LlamaVocab wrapper = LlamaVocab.__new__(LlamaVocab)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @property
    def vocab_type(self) -> llama_vocab_type:
        return int(llama.llama_get_vocab_type(self.ptr))

    @property
    def n_vocab(self) -> int:
        return llama.llama_vocab_n_tokens(self.ptr)

    def get_text(self, llama.llama_token token) -> str:
        return llama.llama_vocab_get_text(self.ptr, token).decode("utf-8")

    def get_score(self, llama.llama_token token) -> float:
        return llama.llama_vocab_get_score(self.ptr, token)

    def get_attr(self, llama.llama_token token) -> llama.llama_token_attr:
        return llama.llama_vocab_get_attr(self.ptr, token)

    def is_eog(self, llama.llama_token token) -> bool:
        """Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)"""
        return llama.llama_vocab_is_eog(self.ptr, token)

    def is_control(self, llama.llama_token token) -> bool:
        """Identify if Token Id is a control token or a render-able token"""
        return llama.llama_vocab_is_control(self.ptr, token)

    # Special tokens

    def token_bos(self) -> int:
        """beginning-of-sentence"""
        return llama.llama_vocab_bos(self.ptr)

    def token_eos(self) -> int:
        """end-of-sentence"""
        return llama.llama_vocab_eos(self.ptr)

    def token_eot(self) -> int:
        """end-of-turn"""
        return llama.llama_vocab_eot(self.ptr)

    def token_sep(self) -> int:
        """sentence separator"""
        return llama.llama_vocab_sep(self.ptr)

    def token_nl(self) -> int:
        """next-line"""
        return llama.llama_vocab_nl(self.ptr)

    def token_pad(self) -> int:
        """padding"""
        return llama.llama_vocab_pad(self.ptr)

    def get_add_bos(self) -> bool:
        """add beginning-of-sentence token"""
        return llama.llama_vocab_get_add_bos(self.ptr)

    def get_add_eos(self) -> bool:
        """add end-of-sentence token"""
        return llama.llama_vocab_get_add_eos(self.ptr)

    def get_add_sep(self) -> bool:
        """add separator token"""
        return llama.llama_vocab_get_add_sep(self.ptr)

    # infill tokens

    def fim_prefix(self) -> int:
        return llama.llama_vocab_fim_pre(self.ptr)

    def fim_middle(self) -> int:
        return llama.llama_vocab_fim_suf(self.ptr)

    def fim_suffix(self) -> int:
        return llama.llama_vocab_fim_mid(self.ptr)

    def fim_pad(self) -> int:
        return llama.llama_vocab_fim_pad(self.ptr)

    def fim_rep(self) -> int:
        return llama.llama_vocab_fim_rep(self.ptr)

    def fim_sep(self) -> int:
        return llama.llama_vocab_fim_sep(self.ptr)

    # Tokenization

    def tokenize(self, text: str, add_special: bool, parse_special: bool) -> list[int]:
        """Convert the provided text into tokens.

        text: string to be converted into token.
        add_special: Allow to add BOS and EOS tokens if model is configured to do so.
        parse_special: Allow tokenizing special and/or control tokens which otherwise
                       are not exposed and treated as plaintext. Does not insert a leading space.
        Returns the number of tokens on success, no more than n_tokens_max
        Returns a negative number on failure - the number of tokens that would have been returned
        """
        # Pre-allocate with reasonable maximum (most texts are much shorter than vocab size)
        cdef int max_tokens = min(self.n_vocab, len(text) * 2 + 100)  # Conservative estimate
        cdef llama.llama_token * tokens = <llama.llama_token *>malloc(sizeof(llama.llama_token) * max_tokens)
        cdef bytes text_bytes = text.encode()
        cdef int text_len = len(text_bytes)
        cdef const char* text_ptr = <const char*>text_bytes
        cdef int n_tokens
        cdef int i

        # Call llama_tokenize - optimization: reduced memory allocation overhead
        n_tokens = llama.llama_tokenize(
            self.ptr, text_ptr, text_len, tokens, max_tokens, add_special, parse_special
        )

        if n_tokens < 0:
            free(tokens)
            raise RuntimeError(
                f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
            )

        # Pre-allocate result list with known size for better performance
        cdef list[int] _tokens = [0] * n_tokens  # Pre-allocate with correct size

        # Optimized token copying loop - direct assignment instead of append
        for i in range(n_tokens):
            _tokens[i] = tokens[i]

        free(tokens)
        return _tokens

    def token_to_piece(self, token: int, lstrip: int = 0, special: bool = False) -> str:
        """Token Id -> Piece.

        special: If true, special tokens are rendered in the output.
        Uses the vocabulary in the provided context.
        Does not write null terminator to the buffer.
        User can skip up to 'lstrip' leading spaces before copying
        (useful when encoding/decoding multiple tokens with 'add_space_prefix')
        """
        cdef char buf[128]
        cdef int32_t length = llama.llama_token_to_piece(self.ptr, token, buf, 128, lstrip, special)
        if length < 0:
            raise ValueError(f"Failed to convert token {token} to piece")
        return buf[:length].decode("utf-8", errors="replace")

    def detokenize(self, tokens: list[int], text_len_max: int = 1024, remove_special: bool = False, unparse_special: bool = False) -> str:
        """Convert the provided tokens into text (inverse of llama_tokenize()).

        @param text The char pointer must be large enough to hold the resulting text.
        @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
        @param unparse_special If true, special tokens are rendered in the output.

        @return Returns the number of chars/bytes on success, no more than text_len_max.
        @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
        """
        cdef str result = ""
        cdef char * buf = <char *>malloc(sizeof(char) * text_len_max)
        cdef std_vector[int] vec

        for i in tokens:
            vec.push_back(i)

        cdef int32_t res = llama.llama_detokenize(
            self.ptr,
            <const llama.llama_token *>vec.data(),
            vec.size(),
            buf,
            text_len_max,
            remove_special,
            unparse_special)

        if res < 0:
            raise RuntimeError(
                f'Failed to detokenize: text="{res}" n_tokens={vec.size()}'
            )
        result = buf.decode()
        free(buf)
        return result.lstrip()


cdef class LlamaModel:
    """cython wrapper for llama.llama_model."""
    cdef llama.llama_model * ptr
    cdef public LlamaModelParams params
    cdef public str path_model
    cdef public bint verbose
    cdef bint owner

    # Cached property values for performance optimization
    cdef int _cached_n_embd
    cdef int _cached_n_layer
    cdef int _cached_n_head
    cdef int _cached_n_head_kv
    cdef int _cached_n_ctx_train
    cdef uint64_t _cached_n_params
    cdef uint64_t _cached_size
    cdef bint _cache_initialized

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True
        self._cache_initialized = False
        # Initialize cached values to -1/0 to indicate they need to be computed
        self._cached_n_embd = -1
        self._cached_n_layer = -1
        self._cached_n_head = -1
        self._cached_n_head_kv = -1
        self._cached_n_ctx_train = -1
        self._cached_n_params = 0
        self._cached_size = 0

    def __init__(self, path_model: str, params: Optional[LlamaModelParams] = None, verbose: bool = True):
        self.path_model = path_model
        self.params = params if params else LlamaModelParams()
        self.verbose = verbose

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        # with suppress_stdout_stderr(disable=verbose):
        self.ptr = llama.llama_model_load_from_file(
            self.path_model.encode("utf-8"),
            self.params.p
        )

        if self.ptr is NULL:
            raise ValueError(f"Failed to load model from file: {path_model}")

        # Initialize property cache after model is loaded
        self._initialize_cache()

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama.llama_model_free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaModel from_ptr(llama.llama_model *ptr, bint owner=False):
        cdef LlamaModel wrapper = LlamaModel.__new__(LlamaModel)
        wrapper.ptr = ptr
        wrapper.owner = owner
        if ptr is not NULL:
            wrapper._initialize_cache()
        return wrapper

    cdef void _initialize_cache(self):
        """Initialize cached property values for performance optimization."""
        if self.ptr is not NULL and not self._cache_initialized:
            self._cached_n_embd = llama.llama_model_n_embd(self.ptr)
            self._cached_n_layer = llama.llama_model_n_layer(self.ptr)
            self._cached_n_head = llama.llama_model_n_head(self.ptr)
            self._cached_n_head_kv = llama.llama_model_n_head_kv(self.ptr)
            self._cached_n_ctx_train = llama.llama_model_n_ctx_train(self.ptr)
            self._cached_n_params = llama.llama_model_n_params(self.ptr)
            self._cached_size = llama.llama_model_size(self.ptr)
            self._cache_initialized = True

    @property
    def rope_type(self) -> llama_rope_type:
        return int(llama.llama_get_model_rope_type(self.ptr))

    @property
    def n_ctx_train(self) -> int:
        if self._cache_initialized:
            return self._cached_n_ctx_train
        return llama.llama_model_n_ctx_train(self.ptr)

    @property
    def n_embd(self) -> int:
        if self._cache_initialized:
            return self._cached_n_embd
        return llama.llama_model_n_embd(self.ptr)

    @property
    def n_layer(self) -> int:
        if self._cache_initialized:
            return self._cached_n_layer
        return llama.llama_model_n_layer(self.ptr)

    @property
    def n_head(self) -> int:
        if self._cache_initialized:
            return self._cached_n_head
        return llama.llama_model_n_head(self.ptr)

    @property
    def n_head_kv(self) -> int:
        if self._cache_initialized:
            return self._cached_n_head_kv
        return llama.llama_model_n_head_kv(self.ptr)

    @property
    def rope_freq_scale_train(self) -> float:
        """Get the model's RoPE frequency scaling factor"""
        return llama.llama_model_rope_freq_scale_train(self.ptr)

    @property
    def desc(self) -> str:
        """Get a string describing the model type"""
        cdef char buf[1024]
        llama.llama_model_desc(self.ptr, buf, 1024)
        return buf.decode("utf-8")

    @property
    def size(self) -> int:
        """Returns the total size of all the tensors in the model in bytes"""
        if self._cache_initialized:
            return self._cached_size
        return <uint64_t>llama.llama_model_size(self.ptr)

    @property
    def n_params(self) -> int:
        """Returns the total number of parameters in the model"""
        if self._cache_initialized:
            return self._cached_n_params
        return <uint64_t>llama.llama_model_n_params(self.ptr)

    # def get_tensor(self, name: str) -> GgmlTensor:
    #     """Get a llama model tensor"""
    #     cdef llama.ggml_tensor * tensor = llama.llama_get_model_tensor(
    #         self.ptr, name.encode("utf-8"))
    #     return GgmlTensor.from_ptr(tensor)

    # vocab

    def get_vocab(self) -> LlamaVocab:
        cdef llama.llama_vocab * vocab = <llama.llama_vocab *>llama.llama_model_get_vocab(self.ptr)
        return LlamaVocab.from_ptr(vocab)

    # sampling

    # def sampler_init(self, CommonParamsSampling params) -> CommonSampler:
    #     """initialize common_sampler"""
    #     return CommonSampler(self, params)

    # lora

    def lora_adapter_init(self, str path_lora) -> LlamaAdapterLora:
        """Load a LoRA adapter from file

        The loaded adapter will be associated to the given model, and will be free when the model is deleted
        """
        cdef llama.llama_adapter_lora * ptr = llama.llama_adapter_lora_init(
            self.ptr, path_lora.encode())
        return LlamaAdapterLora.from_ptr(ptr)
 
    # metadata

    def meta_val_str(self, str key) -> str:
        """Get metadata value as a string by key name"""
        cdef char buf[128]
        cdef int res = llama.llama_model_meta_val_str(self.ptr, key.encode(), buf, 128)
        if res == -1:
            raise ValueError(F"could not get metadata value from {key}")
        cdef str value = buf.decode('UTF-8')
        return value

    def meta_count(self):
        """Get the number of metadata key/value pairs"""
        return llama.llama_model_meta_count(self.ptr)

    def meta_key_by_index(self, int index) -> str:
        """Get metadata key name by index"""
        cdef char buf[128]
        cdef int res = llama.llama_model_meta_key_by_index(self.ptr, index, buf, 128)
        cdef str key = buf.decode('UTF-8')
        return key

    def meta_val_str_by_index(self, int index) -> str:
        """Get metadata key name by index"""
        cdef char buf[128]
        cdef int res = llama.llama_model_meta_val_str_by_index(self.ptr, index, buf, 128)
        cdef str value = buf.decode('UTF-8')
        return value

    # encode / decode

    def has_encoder(self) -> bool:
        """Returns true if the model contains an encoder that requires llama_encode() call"""
        return llama.llama_model_has_encoder(self.ptr)

    def has_decoder(self) -> bool:
        """Returns true if the model contains a decoder that requires llama_decode() callD"""
        return llama.llama_model_has_decoder(self.ptr)

    def decoder_start_token(self) -> int:
        """For encoder-decoder models, this function returns id of the token that must be provided
        to the decoder to start generating output sequence. For other models, it returns -1.
        """
        return llama.llama_model_decoder_start_token(self.ptr)

    def is_recurrent(self) -> bool:
        """Returns true if the model is recurrent (like Mamba, RWKV, etc.)"""
        return llama.llama_model_is_recurrent(self.ptr)

    # chat template

    def chat_apply_template(self, str tmpl, list[LlamaChatMessage] msgs, bint add_assistant_msg) -> str:
        """Apply chat template. Inspired by hf apply_chat_template() on python.

        Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
        NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
        @param tmpl A Jinja template to use for this chat. If this is nullptr, the model's default chat template will be used instead.
        @param chat Pointer to a list of multiple llama_chat_message
        @param n_msg Number of llama_chat_message in this chat
        @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
        @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
        @param length The size of the allocated buffer
        @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
        """
        cdef std_vector[llama.llama_chat_message] vec
        cdef LlamaChatMessage msg
        vec.reserve(len(msgs))
        for i in range(len(msgs)):
            msg = msgs[i]
            vec.push_back(msg.p)
        
        # First call to get required buffer size
        cdef const char* tmpl_ptr = NULL
        cdef bytes tmpl_bytes
        if tmpl is not None:
            tmpl_bytes = tmpl.encode()
            tmpl_ptr = tmpl_bytes
        
        cdef int32_t required_size = llama.llama_chat_apply_template(
            tmpl_ptr,
            vec.data(),
            vec.size(),
            add_assistant_msg,
            NULL,
            0
        )
        
        if required_size < 0:
            raise RuntimeError("Failed to apply chat template")
        
        # Allocate buffer and apply template
        cdef char* buf = <char*>malloc(sizeof(char) * (required_size + 1))
        if buf is NULL:
            raise MemoryError("Failed to allocate buffer for chat template")
        
        cdef int32_t actual_size = llama.llama_chat_apply_template(
            tmpl_ptr,
            vec.data(),
            vec.size(),
            add_assistant_msg,
            buf,
            required_size
        )
        
        if actual_size < 0:
            free(buf)
            raise RuntimeError("Failed to apply chat template")
        
        # Ensure null termination
        buf[actual_size] = 0
        cdef str result = buf[:actual_size].decode("utf-8")
        free(buf)
        return result

    def get_default_chat_template(self) -> str:
        """Get the default chat template for the model.

        Return empty string if not present.
        """
        cdef const char * res = llama.llama_model_chat_template(self.ptr, NULL)
        if res:
            return res.decode()
        return ""

    def get_default_chat_template_by_name(self, str name) -> str:
        """Get the default chat template for the model by name

        Return empty string if not present.
        """
        cdef const char * res = llama.llama_model_chat_template(self.ptr, name.encode())
        if res:
            return res.decode()
        return ""

    # Extra

    def metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        buffer_size = 1024
        cdef int nbytes
        cdef char * buffer = <char*>calloc(buffer_size, sizeof(char))
        assert self.ptr is not NULL
        # iterate over model keys
        for i in range(llama.llama_model_meta_count(self.ptr)):
            nbytes = llama.llama_model_meta_key_by_index(
                self.ptr, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama.llama_model_meta_key_by_index(
                    self.ptr, i, buffer, buffer_size
                )
            key = buffer.decode("utf-8")
            nbytes = llama.llama_model_meta_val_str_by_index(
                self.ptr, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama.llama_model_meta_val_str_by_index(
                    self.ptr, i, buffer, buffer_size
                )
            value = buffer.decode("utf-8")
            metadata[key] = value
        free(buffer)
        return metadata

    @staticmethod
    def default_params() -> LlamaModelParams:
        """Get the default llama_model_params."""
        # return llama.llama_model_default_params()
        return LlamaModelParams()


cdef class LlamaContext:
    """Intermediate Python wrapper for a llama.cpp llama_context."""
    cdef llama.llama_context * ptr
    cdef public LlamaModel model
    cdef public LlamaContextParams params
    cdef public bint verbose
    cdef public int n_tokens
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True
        self.n_tokens = 0

    def __init__(self, model: LlamaModel, params: Optional[LlamaContextParams] = None, verbose: bool = True):
        self.model = model
        self.params = params if params else LlamaContextParams()
        self.verbose = verbose

        # self.ptr = None

        assert self.model.ptr is not NULL

        self.ptr = llama.llama_init_from_model(self.model.ptr, self.params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to create llama_context")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama.llama_free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaContext from_ptr(llama.llama_context *ptr, bint owner=False):
        cdef LlamaContext wrapper = LlamaContext.__new__(LlamaContext)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    def close(self):
        self.__dealloc__()

    @property
    def n_ctx(self) -> int:
        return llama.llama_n_ctx(self.ptr)

    @property
    def n_batch(self) -> int:
        return llama.llama_n_batch(self.ptr)

    @property
    def n_ubatch(self) -> int:
        return llama.llama_n_ubatch(self.ptr)

    @property
    def n_seq_max(self) -> int:
        return llama.llama_n_seq_max(self.ptr)

    @property
    def pooling_type(self) -> int:
        return <llama.llama_pooling_type>llama.llama_get_pooling_type(self.ptr)

    # Manage Threadpools
    # -------------------------------------------------------------------------

    def attach_threadpool(self, GgmlThreadPool threadpool, GgmlThreadPool threadpool_batch):
        llama.llama_attach_threadpool(self.ptr, threadpool.ptr, threadpool_batch.ptr)

    def detach_threadpool(self):
        llama.llama_detach_threadpool(self.ptr)

    # State / sessions
    # -------------------------------------------------------------------------

    def get_state_size(self) -> int:
        """Returns the *actual* size in bytes of the state

        (logits, embedding and kv_cache)
        Only use when saving the state, not when restoring it, otherwise the size may be too small.
        """
        return llama.llama_state_get_size(self.ptr)

    def get_state_data(self) -> list[int]:
        """Copies the state to the specified destination address.

        Destination needs to have allocated enough memory.
        Returns the number of bytes copied
        """
        cdef uint8_t * dst = NULL
        cdef size_t size = 0
        cdef std_vector[uint8_t] result
        cdef size_t copied = llama.llama_state_get_data(self.ptr, dst, size)
        for i in range(size):
            result.push_back(dst[i])
        return result

    def set_state_data(self, data: list[int]) -> int:
        """Set the state reading from the specified address

        Returns the number of bytes read
        """
        cdef std_vector[uint8_t] result = data
        cdef size_t read = llama.llama_state_set_data(self.ptr, result.data(), result.size())
        return read

    def load_state_file(self, path_session: str, max_n_tokens: int = 256) -> list[int]:
        """Load session file"""
        cdef llama.llama_token * tokens_out = NULL
        cdef size_t * n_token_count_out = NULL
        cdef bint loaded = llama.llama_state_load_file(
            self.ptr,
            path_session.encode(),
            tokens_out,
            max_n_tokens,
            n_token_count_out)
        cdef std_vector[int] result
        if loaded:
            for i in range(n_token_count_out[0]):
                result.push_back(tokens_out[i])
        return result

    def save_state_file(self, path_session: str, tokens: list[int]) -> bool:
        """Save session file"""
        cdef std_vector[llama.llama_token] vec_tokens
        for token in tokens:
            vec_tokens.push_back(<llama.llama_token>token)
        return llama.llama_state_save_file(
            self.ptr,
            path_session.encode(),
            vec_tokens.data(),
            vec_tokens.size())

    def get_state_seq_size(self, int seq_id) -> int:
        """Get the exact size needed to copy the KV cache of a single sequence"""
        return llama.llama_state_seq_get_size(self.ptr, seq_id)

    def get_state_seq_data(self, int seq_id) -> list[int]:
        """Copy the KV cache of a single sequence into the specified buffer"""
        cdef uint8_t dst[512];
        cdef size_t copied = llama.llama_state_seq_get_data(
            self.ptr, dst, 512, seq_id)
        cdef std_vector[uint8_t] result
        for i in range(copied):
            result.push_back(dst[i])
        return result

    def set_state_seq_data(self, src: list[int], dest_seq_id: int):
        """Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence

        Returns:
         - Positive: Ok
         - Zero: Failed to load
        """
        cdef std_vector[uint8_t] vec
        cdef size_t res = 0
        for i in src:
            vec.push_back(i)
        res = llama.llama_state_seq_set_data(
            self.ptr, vec.data(), vec.size(), dest_seq_id)
        if res == 0:
            raise ValueError("Failed to load sequence data")

    def save_state_seq_file(self, filepath: str, seq_id: int, tokens: list[int]):
        """Save state sequence data to a file"""
        cdef std_vector[uint8_t] vec
        cdef size_t res = 0
        for i in tokens:
            vec.push_back(i)
        res = llama.llama_state_seq_save_file(
            self.ptr,
            filepath.encode(),
            seq_id,
            <const llama.llama_token *>vec.data(),
            vec.size())
        if res == 0:
            raise ValueError(f"Failed to save seq data {filepath}")

    def load_state_seq_file(self, filepath: str, dest_seq_id: int, max_n_tokens: int = 256):
        """Load state sequence data from a file"""
        cdef llama.llama_token * tokens_out = NULL
        cdef size_t * n_token_count_out = NULL
        cdef size_t loaded = llama.llama_state_seq_load_file(
            self.ptr,
            filepath.encode(),
            dest_seq_id,
            tokens_out,
            max_n_tokens,
            n_token_count_out)
        cdef std_vector[int] result
        if loaded:
            for i in range(n_token_count_out[0]):
                result.push_back(tokens_out[i])
        return result

    def get_state_seq_size_with_flags(self, int seq_id, int flags) -> int:
        """get state sequence size from seq_id and flags"""
        return llama.llama_state_seq_get_size_ext(self.ptr, seq_id, flags)

    def get_state_seq_data_with_flags(self, int seq_id, int flags) -> list[int]:
        """get state sequence daya from seq_id and flags"""
        cdef uint8_t dst[512];
        cdef size_t size = llama.llama_state_seq_get_data_ext(self.ptr, dst, 512, seq_id, flags)
        cdef std_vector[uint8_t] result
        for i in range(size):
            result.push_back(dst[i])
        return result

    def set_state_seq_data_with_flags(self, src: list[int], dest_seq_id: int,  flags: int):
        """set state seq data with flags"""
        cdef std_vector[uint8_t] vec
        cdef size_t res = 0
        for i in src:
            vec.push_back(i)
        res = llama.llama_state_seq_set_data_ext(
            self.ptr, vec.data(), vec.size(), dest_seq_id, flags)
        if res == 0:
            raise ValueError("Failed to set sequence data")


    # Decoding
    # -------------------------------------------------------------------------

    def encode(self, LlamaBatch batch):
        """Processes a batch of tokens with the encoder part of the encoder-decoder model.

        Stores the encoder output internally for later use by the decoder cross-attention layers.
          0 - success
        < 0 - error
        """
        cdef int32_t res = llama.llama_encode(self.ptr, batch.p)
        if res < 0:
            raise RuntimeError("error encoding batch")

    def decode(self, LlamaBatch batch) -> int:
        """Positive return values does not mean a fatal error, but rather a warning.

          0 - success
          1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
        < 0 - error
        """
        cdef int32_t res = llama.llama_decode(self.ptr, batch.p)
        self.n_tokens = batch.n_tokens
        if res == 1:
            raise ValueError("could not find a KV slot for the batch (try reducing the size of the batch or increase the context)")
        if res < 0:
            raise RuntimeError(f"llama_decode failed")
        return res

    def set_n_threads(self, n_threads: int, n_threads_batch: int):
        """Set the number of threads used for decoding

        n_threads is the number of threads used for generation (single token)
        n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
        """
        llama.llama_set_n_threads(self.ptr, n_threads, n_threads_batch)

    def n_threads(self):
        """Get the number of threads used for generation of a single token."""
        return llama.llama_n_threads(self.ptr)

    def n_threads_batch(self):
        """Get the number of threads used for prompt and batch processing (multiple token)."""
        return llama.llama_n_threads_batch(self.ptr)

    def set_embeddings_mode(self, embeddings: bool):
        """Set whether the model is in embeddings mode or not

        If true, embeddings will be returned but logits will not
        """
        llama.llama_set_embeddings(self.ptr, embeddings)

    def set_causal_attn(self, causal_attn: bool):
        """Set whether to use causal attention or not

        If set to true, the model will only attend to the past tokens
        """
        llama.llama_set_causal_attn(self.ptr, causal_attn)

    def set_abort_callback(self, object py_abort_callback):
        """Set abort callback"""
        llama.llama_set_abort_callback(self.ptr,
            <ggml.ggml_abort_callback>&abort_callback, <void*>py_abort_callback)

    def synchronize(self):
        """Wait until all computations are finished

        This is automatically done when using one of the functions below to obtain the computation results
        and is not necessary to call it explicitly in most cases
        """
        llama.llama_synchronize(self.ptr)

    # def n_outputs(self) -> int:
    #     return llama.llama_n_outputs(self.ptr)

    def get_logits(self) -> list[float]:
        """Token logits obtained from the last call to llama_decode()

        The logits for which llama_batch.logits[i] != 0 are stored contiguously
        in the order they have appeared in the batch.

        Rows: number of tokens for which llama_batch.logits[i] != 0
        Cols: n_vocab
        """
        cdef int n_vocab = self.model.n_vocab
        cdef float * logits = llama.llama_get_logits(self.ptr)
        if logits is NULL:
            # TODO: should one just return [] here?
            raise ValueError('no logits available')
        cdef std_vector[float] vec
        for i in range(n_vocab):
            vec.push_back(logits[i])
        return vec

    def get_logits_ith(self, int i):
        """Logits for the ith token. For positive indices,

        Equivalent to:
        llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
        Negative indicies can be used to access logits in reverse order, -1 is the last logit.
        returns NULL for invalid ids.
        """
        cdef int n_vocab = self.model.n_vocab
        cdef float * logits = llama.llama_get_logits_ith(self.ptr, i)
        cdef std_vector[float] vec
        if logits is NULL:
            raise ValueError(f"{i} is an invalid id")
        for i in range(n_vocab):
            vec.push_back(logits[i])
        return vec

    def get_embeddings(self):
        """Get all output token embeddings.

        when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
        the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
        in the order they have appeared in the batch.
        shape: [n_outputs * n_embd]
        Otherwise, returns NULL.
        """
        cdef int n_embd = self.model.n_embd
        cdef float * embds = llama.llama_get_embeddings(self.ptr)
        cdef std_vector[float] vec
        if embds is NULL:
            # TODO: should one just return [] here?
            raise ValueError('no embeddings available')
        for i in range(n_embd):
            vec.push_back(embds[i])
        return vec

    def get_embeddings_ith(self, int i):
        """Get the embeddings for the ith token. For positive indices, Equivalent to:

        llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
        Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
        returns NULL for invalid ids.
        """
        cdef int n_embd = self.model.n_embd
        cdef float * embds = llama.llama_get_embeddings_ith(self.ptr, i)
        cdef std_vector[float] vec
        if embds is NULL:
            raise ValueError(f"{i} is an invalid id")
        for i in range(n_embd):
            vec.push_back(embds[i])
        return vec

    # def get_embeddings_seq(self, int seq_id):
    #     """Get the embeddings for a sequence id

    #     Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    #     when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
    #     otherwise: float[n_embd] (1-dimensional)
    #     """
    #     cdef float * embds = llama_get_embeddings_seq(self.ptr, seq_id)

    # Utility functions
    @staticmethod
    def default_params():
        """Get the default llama_context_params."""
        return LlamaContext()

    #
    # Performance utils
    #
    # NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.
    #

    # def get_perf_data(self):
    #     """get performance data"""
    #     cdef llama_perf_context_data data = llama.llama_perf_context(self.ptr)

    def print_perf_data(self):
        """print performance data"""
        llama.llama_perf_context_print(self.ptr)

    def reset_perf_data(self):
        """reset performance data"""
        llama.llama_perf_context_reset(self.ptr)

    def print_memory_breakdown(self):
        """print a breakdown of per-device memory use via LLAMA_LOG"""
        llama.llama_memory_breakdown_print(self.ptr)


cdef class LlamaSampler:
    """cython wrapper for llama.llama_sampler."""
    cdef llama.llama_sampler * ptr
    cdef LlamaSamplerChainParams params
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, params: Optional[LlamaSamplerChainParams] = None):
        if not params:
            self.ptr = llama.llama_sampler_chain_init(self.params.p)
        else:
            self.ptr = llama.llama_sampler_chain_init(params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to init Sampler")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama.llama_sampler_free(self.ptr)
            self.ptr = NULL

    def name(self) -> str:
        """Get sampler name"""
        return llama.llama_sampler_name(self.ptr).decode()

    def accept(self, llama.llama_token token):
        """Accept llama token"""
        llama.llama_sampler_accept(self.ptr, token)

    # cdef void llama_sampler_apply (llama_sampler * smpl, llama_token_data_array * cur_p)

    def reset(self):
        """Reset sampler"""
        llama.llama_sampler_reset(self.ptr)

    def clone(self) -> LlamaSampler:
        """clone sampler"""
        cdef llama.llama_sampler * smplr = llama.llama_sampler_clone(self.ptr)
        cdef LlamaSampler wrapper = LlamaSampler.__new__(LlamaSampler)
        wrapper.ptr = smplr
        return wrapper

    def get_seed(self) -> int:
        """Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise"""
        return llama.llama_sampler_get_seed(self.ptr)

    def add_greedy(self):
        """Add greedy sampling chain link

        This should be at the end of the chain.
        """
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_greedy())

    def add_dist(self, uint32_t seed):
        """Add dist sampling chain link

        This should be at the end of the chain.
        """
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_dist(seed))

    # DEPRECATED
    # def add_softmax(self):
    #     """Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits."""
    #     llama.llama_sampler_chain_add(
    #         self.ptr, llama.llama_sampler_init_softmax())

    def add_top_k(self, int32_t k):
        """Add Top-K sampling chain link.

        Described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751"""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_top_k(k))

    def add_top_p(self, float p, size_t min_keep):
        """Add Nucleus sampling chain link.

        Described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751"""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_top_p(p, min_keep))

    def add_min_p(self, float p, size_t min_keep):
        """Add Minimum P sampling.

        Described in https:#github.com/ggerganov/llama.cpp/pull/3841"""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_min_p(p, min_keep))

    def add_typical(self, float p, size_t min_keep):
        """Add Locally Typical Sampling implementation.

        Described in the paper https:#arxiv.org/abs/2202.00666."""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_typical(p, min_keep))

    def add_temp(self, float t):
        """Add temperature sampling chain link.

        Updates the logits `l_i = l_i/t`. When `t <= 0.0f`,
        the maximum logit is kept at its original value, the rest are set to -inf."""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_temp(t))

    def add_temp_ext(self, float t, float delta, float exponent):
        """Add Dynamic temperature implementation sampling chain link

        Described in the paper https:#arxiv.org/abs/2309.02772."""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_temp_ext(t, delta, exponent))

    def add_xtc(self, float p, float t, size_t min_keep, uint32_t seed):
        """Add XTC sampler chain link

        Described in https://github.com/oobabooga/text-generation-webui/pull/6335"""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_xtc(p, t, min_keep, seed))

    # XXX: docstring incorrect
    def add_mirostat(self, int n_vocab, uint32_t seed, float tau, float eta, int m):
        """Mirostat 1.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.

        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau:     The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta:     The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        m:       The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
        mu:      Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
        """
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m))

    def add_mirostat_v2(self, uint32_t seed, float tau, float eta):
        """Mirostat 2.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.

        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau:  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        mu: Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
        """
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_mirostat_v2(seed, tau, eta))

    def add_grammar(self, LlamaVocab vocab, str grammar_str, str grammar_root):
        """Add grammer chain link"""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_grammar(
                vocab.ptr, grammar_str.encode(), grammar_root.encode()))

    def add_penalties(self,
         int penalty_last_n,   # last n tokens to penalize (0 = disable penalty, -1 = context size)
       float penalty_repeat,   # 1.0 = disabled
       float penalty_freq,     # 0.0 = disabled
       float penalty_present): # 0.0 = disabled
        """Add penalties chain link"""
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_penalties(
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            ))

    # XXX FIXME:
    # def add_logit_bias(self, int n_vocab, int n_logit_bias, logit_bias: list[LogitBias]):
    #     """Add grammer chain link"""
    #     cdef std_vector[llama.logit_bias] vec
    #     llama.llama_sampler_chain_add(
    #         self.ptr, llama.llama_sampler_init_logit_bias(
    #             n_vocab, n_logit_bias, vec.data()))

    def add_infill(self, LlamaVocab vocab):
        """This sampler is meant to be used for fill-in-the-middle infilling

        it's supposed to be used after top_k + top_p sampling

        1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
        2. combine probs of tokens that have the same prefix

        example:

        - before:
          "hel":   0.5
          "hell":  0.2
          "hello": 0.1
          "dummy": 0.1

        - after:
          "hel":   0.8
          "dummy": 0.1

        3. discard non-EOG tokens with low prob
        4. if no tokens are left -> pick EOT
        """
        llama.llama_sampler_chain_add(
            self.ptr,
            llama.llama_sampler_init_infill(vocab.ptr)
        )


    def sample(self, LlamaContext ctx, int idx) -> int:
        """Sample and accept a token from the idx-th output of the last evaluation

        Shorthand for:

           const auto * logits = llama_get_logits_ith(ctx, idx)
           llama_token_data_array cur_p = { ... init from logits ... }
           llama_sampler_apply(smpl, &cur_p)
           return cur_p.data[cur_p.selected].id

        At this point, this is mostly a convenience function.
        """
        return llama.llama_sampler_sample(self.ptr, ctx.ptr, idx)


    #
    # Performance utils
    #
    # NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.
    #


    # def get_perf_data(self):
    #     """
    #     # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    #     """
    #     cdef llama_perf_sampler_data data = llama.llama_perf_sampler(self.ptr)

    def print_perf_data(self):
        """
        # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
        """
        llama.llama_perf_sampler_print(self.ptr)

    def reset_perf_data(self):
        """
        # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
        """
        llama.llama_perf_sampler_reset(self.ptr)




cdef class LlamaAdapterLora:
    cdef llama.llama_adapter_lora * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            llama.llama_adapter_lora_free(self.ptr)
            self.ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef LlamaAdapterLora from_ptr(llama.llama_adapter_lora *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef LlamaAdapterLora wrapper = LlamaAdapterLora.__new__(LlamaAdapterLora)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    # Functions to access the adapter's GGUF metadata scalar values
    # - The functions return the length of the string on success, or -1 on failure
    # - The output string is always null-terminated and cleared on failure
    # - When retrieving a string, an extra byte must be allocated to account for the null terminator
    # - GGUF array values are not supported by these functions

    def meta_val_str(self, str key) -> str:
        """Get metadata value as a string by key name"""
        assert key != "", "key must not be an empty string"
        cdef char buf[512]
        cdef int32_t str_len = llama.llama_adapter_meta_val_str(self.ptr, key.encode(), buf, 512)
        if str_len == -1:
            raise ValueError("failed to retrieve metadata value")
            # TODO: log.debug str_len 
        return buf.decode()

    def meta_count(self) -> int:
        """Get the number of metadata key/value pairs"""
        return llama.llama_adapter_meta_count(self.ptr)

    def meta_key_by_index(self, int idx = 0) -> str:
        """Get metadata key name by index"""
        cdef char buf[512]
        cdef int32_t str_len = llama.llama_adapter_meta_key_by_index(self.ptr, idx, buf, 512)
        if str_len == -1:
            raise ValueError("failed to retrieve metadata key")
        return buf.decode()

    def meta_val_str_by_index(self, int idx = 0) -> str:
        """Get metadata value as a string by index"""
        cdef char buf[512]
        cdef int32_t str_len = llama.llama_adapter_meta_val_str_by_index(self.ptr, idx, buf, 512)
        if str_len == -1:
            raise ValueError("failed to retrieve metadata value")
        return buf.decode()

# -------------------------------------------------------------------------
# functions

cdef void no_log_cb(ggml.ggml_log_level l, const char * x, void * d) noexcept:
    pass

def disable_logging():
    llama.llama_log_set(no_log_cb, NULL)

def chat_builtin_templates() -> list[str]:
    """Get list of built-in chat templates"""
    cdef std_vector[const char *] supported_tmpl
    cdef int32_t res = llama.llama_chat_builtin_templates(NULL, 0)
    assert res > 0
    supported_tmpl.resize(res)
    res = llama.llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size())
    return [name.decode() for name in supported_tmpl]

def ggml_version() -> str:
    return ggml.ggml_version().decode()

def ggml_commit() -> str:
    return ggml.ggml_commit().decode()

def ggml_backend_load_all():
    ggml.ggml_backend_load_all()

def ggml_time_us() -> int:
    return ggml.ggml_time_us()

def llama_backend_init():
    """Initialize the llama + ggml backend

    If numa is true, use NUMA optimizations
    Call once at the start of the program
    """
    llama.llama_backend_init()

def llama_numa_init(ggml.ggml_numa_strategy numa):
    llama.llama_numa_init(numa)

def llama_time_us() -> int:
    return llama.llama_time_us()

def llama_max_devices() -> int:
    return llama.llama_max_devices()

def llama_supports_mmap() -> bool:
    return llama.llama_supports_mmap()

def llama_supports_mlock() -> bool:
    return llama.llama_supports_mlock()

def llama_supports_gpu_offload() -> bool:
    return llama.llama_supports_gpu_offload()

def llama_supports_rpc() -> bool:
    return llama.llama_supports_rpc()

def llama_attach_threadpool(LlamaContext ctx, GgmlThreadPool threadpool, GgmlThreadPool threadpool_batch):
    llama.llama_attach_threadpool(ctx.ptr, threadpool.ptr, threadpool_batch.ptr)

def llama_detach_threadpool(LlamaContext ctx):
    llama.llama_detach_threadpool(ctx.ptr)

def llama_batch_get_one(list[int] tokens, int n_past = 0) -> LlamaBatch:
    """Create a batch using the proper batch API instead of the deprecated llama_batch_get_one"""
    cdef int32_t n_tokens = <int32_t>len(tokens)
    # for i in range(n_tokens):
    #     print(f"tokens[{i}]: {tokens[i]}")
    
    # Create a proper batch using the new API
    batch = LlamaBatch(n_tokens=n_tokens, embd=0, n_seq_max=1)
    batch.set_batch(tokens, n_past=n_past, logits_all=False)
    return batch

def llama_backend_free():
    """Call once at the end of the program - currently only used for MPI"""
    llama.llama_backend_free()

def llama_flash_attn_type_name(llama.llama_flash_attn_type flash_attn_type) -> str:
    return llama.llama_flash_attn_type_name(flash_attn_type).decode()


