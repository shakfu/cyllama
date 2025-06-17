# distutils: language = c++
"""
cyllama: a thin cython wrapper of llama.cpp

classes:
    LlamaLogitBias
    LlamaTokenData
    LlamaTokenDataArray
    LlamaAdapterLora
    GGMLThreadPoolParams
    GGMLThreadPool
    GGMLTensor
    LlamaSamplerChainParams
    LlamaSampler
    LlamaChatMessage
    - CommonChatMsg
    CommonSampler
    CpuParams
    CommonParams
    CommonParamsSampling
    LlamaModelParams
    LlamaModelQuantizeParams
    LlamaModel
    LlamaVocab
    LlamaContextParams
    LlamaContext
    LlamaBatch
    CommonInitResult


"""
from libc.stdint cimport uint8_t, int32_t, int64_t, uint32_t, uint64_t
from libc.stdlib cimport malloc, calloc, realloc, free
from libc.string cimport strcpy, strlen, strncpy
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool as cppbool # required for func pointer sigs

cimport llama_cpp

# import numpy as np

import os
from typing import Optional, Sequence, Callable



# constants
# -----------------------------------------------------------------------------

LLAMA_DEFAULT_SEED = 0xFFFFFFFF

cpdef enum:
    GGML_DEFAULT_N_THREADS = 4
    GGML_MAX_DIMS = 4
    GGML_MAX_N_THREADS = 16
    GGML_MAX_NAME = 64
    GGML_MAX_OP_PARAMS = 64
    GGML_MAX_SRC = 10

cpdef enum:
    GGML_ROPE_TYPE_NEOX   = 2
    GGML_ROPE_TYPE_MROPE  = 8
    GGML_ROPE_TYPE_VISION = 24


# build info
# -----------------------------------------------------------------------------

BUILD_INFO = {
    'build_number': llama_cpp.LLAMA_BUILD_NUMBER,
    'commit': llama_cpp.LLAMA_COMMIT.decode(),
    'compiler': llama_cpp.LLAMA_COMPILER.decode(),
    'build_target': llama_cpp.LLAMA_BUILD_TARGET.decode(),
}

# enums
# -----------------------------------------------------------------------------

cpdef enum ggml_log_level:
    GGML_LOG_LEVEL_NONE  = 0
    GGML_LOG_LEVEL_INFO  = 1
    GGML_LOG_LEVEL_WARN  = 2
    GGML_LOG_LEVEL_ERROR = 3
    GGML_LOG_LEVEL_DEBUG = 4
    GGML_LOG_LEVEL_CONT  = 5

cpdef enum llama_vocab_type:
    LLAMA_VOCAB_TYPE_NONE # For models without vocab
    LLAMA_VOCAB_TYPE_SPM  # LLaMA tokenizer based on byte-level BPE with byte fallback
    LLAMA_VOCAB_TYPE_BPE  # GPT-2 tokenizer based on byte-level BPE
    LLAMA_VOCAB_TYPE_WPM  # BERT tokenizer based on WordPiece
    LLAMA_VOCAB_TYPE_UGM  # T5 tokenizer based on Unigram
    LLAMA_VOCAB_TYPE_RWKV # RWKV tokenizer based on greedy tokenization

cpdef enum llama_rope_type:
    LLAMA_ROPE_TYPE_NONE   = -1
    LLAMA_ROPE_TYPE_NORM   = 0
    LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX
    LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE
    LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION

cpdef enum llama_token_attr:
    LLAMA_TOKEN_ATTR_UNDEFINED    = 0
    LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0
    LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1
    LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2
    LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3 # SPECIAL?
    LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
    LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
    LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
    LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
    LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
    LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9

cpdef enum ggml_numa_strategy:
    GGML_NUMA_STRATEGY_DISABLED   = 0
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1
    GGML_NUMA_STRATEGY_ISOLATE    = 2
    GGML_NUMA_STRATEGY_NUMACTL    = 3
    GGML_NUMA_STRATEGY_MIRROR     = 4
    GGML_NUMA_STRATEGY_COUNT

cpdef enum ggml_type:
    GGML_TYPE_F32     = 0
    GGML_TYPE_F16     = 1
    GGML_TYPE_Q4_0    = 2
    GGML_TYPE_Q4_1    = 3
    # GGML_TYPE_Q4_2 = 4 support has been removed
    # GGML_TYPE_Q4_3 = 5 support has been removed
    GGML_TYPE_Q5_0    = 6
    GGML_TYPE_Q5_1    = 7
    GGML_TYPE_Q8_0    = 8
    GGML_TYPE_Q8_1    = 9
    GGML_TYPE_Q2_K    = 10
    GGML_TYPE_Q3_K    = 11
    GGML_TYPE_Q4_K    = 12
    GGML_TYPE_Q5_K    = 13
    GGML_TYPE_Q6_K    = 14
    GGML_TYPE_Q8_K    = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS  = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S   = 19
    GGML_TYPE_IQ4_NL  = 20
    GGML_TYPE_IQ3_S   = 21
    GGML_TYPE_IQ2_S   = 22
    GGML_TYPE_IQ4_XS  = 23
    GGML_TYPE_I8      = 24
    GGML_TYPE_I16     = 25
    GGML_TYPE_I32     = 26
    GGML_TYPE_I64     = 27
    GGML_TYPE_F64     = 28
    GGML_TYPE_IQ1_M   = 29
    GGML_TYPE_BF16    = 30
    # GGML_TYPE_Q4_0_4_4 = 31 # support has been removed from gguf files
    # GGML_TYPE_Q4_0_4_8 = 32
    # GGML_TYPE_Q4_0_8_8 = 33
    GGML_TYPE_TQ1_0   = 34
    GGML_TYPE_TQ2_0   = 35
    GGML_TYPE_IQ4_NL_4_4 = 36
    # GGML_TYPE_IQ4_NL_4_4 = 36
    # GGML_TYPE_IQ4_NL_4_8 = 37
    # GGML_TYPE_IQ4_NL_8_8 = 38
    GGML_TYPE_COUNT = 39


cpdef enum ggml_sched_priority:
    GGML_SCHED_PRIO_NORMAL
    GGML_SCHED_PRIO_MEDIUM
    GGML_SCHED_PRIO_HIGH
    GGML_SCHED_PRIO_REALTIME

cpdef enum llama_ftype:
    LLAMA_FTYPE_ALL_F32              = 0
    LLAMA_FTYPE_MOSTLY_F16           = 1
    LLAMA_FTYPE_MOSTLY_Q4_0          = 2
    LLAMA_FTYPE_MOSTLY_Q4_1          = 3
    # LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4  # tok_embeddings.weight and output.weight are F16
    # LLAMA_FTYPE_MOSTLY_Q4_2       = 5     # support has been removed
    # LLAMA_FTYPE_MOSTLY_Q4_3       = 6     # support has been removed
    LLAMA_FTYPE_MOSTLY_Q8_0          = 7
    LLAMA_FTYPE_MOSTLY_Q5_0          = 8
    LLAMA_FTYPE_MOSTLY_Q5_1          = 9
    LLAMA_FTYPE_MOSTLY_Q2_K          = 10
    LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11
    LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12
    LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13
    LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14
    LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15
    LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16
    LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17
    LLAMA_FTYPE_MOSTLY_Q6_K          = 18
    LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19
    LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20
    LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21
    LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22
    LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23
    LLAMA_FTYPE_MOSTLY_IQ1_S         = 24
    LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25
    LLAMA_FTYPE_MOSTLY_IQ3_S         = 26
    LLAMA_FTYPE_MOSTLY_IQ3_M         = 27
    LLAMA_FTYPE_MOSTLY_IQ2_S         = 28
    LLAMA_FTYPE_MOSTLY_IQ2_M         = 29
    LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30
    LLAMA_FTYPE_MOSTLY_IQ1_M         = 31
    LLAMA_FTYPE_MOSTLY_BF16          = 32
    # LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, # removed from gguf files, use Q4_0 and runtime repack
    # LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, # removed from gguf files, use Q4_0 and runtime repack
    # LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, # removed from gguf files, use Q4_0 and runtime repack
    LLAMA_FTYPE_MOSTLY_TQ1_0         = 36 # except 1d tensors
    LLAMA_FTYPE_MOSTLY_TQ2_0         = 37 # except 1d tensors
    LLAMA_FTYPE_GUESSED              = 1024

cpdef enum llama_rope_scaling_type:
    LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    LLAMA_ROPE_SCALING_TYPE_NONE        = 0
    LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
    LLAMA_ROPE_SCALING_TYPE_YARN        = 2
    LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3
    LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE

cpdef enum llama_pooling_type:
    LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    LLAMA_POOLING_TYPE_NONE = 0
    LLAMA_POOLING_TYPE_MEAN = 1
    LLAMA_POOLING_TYPE_CLS  = 2
    LLAMA_POOLING_TYPE_LAST = 3
    LLAMA_POOLING_TYPE_RANK = 4 # used by reranking models to attach the classification head to the graph

cpdef enum llama_attention_type:
    LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    LLAMA_ATTENTION_TYPE_CAUSAL      = 0
    LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

cpdef enum llama_split_mode:
    LLAMA_SPLIT_MODE_NONE  = 0
    LLAMA_SPLIT_MODE_LAYER = 1
    LLAMA_SPLIT_MODE_ROW   = 2

cpdef enum common_sampler_type:
    COMMON_SAMPLER_TYPE_NONE        = 1
    COMMON_SAMPLER_TYPE_TOP_K       = 2
    COMMON_SAMPLER_TYPE_TOP_P       = 3
    COMMON_SAMPLER_TYPE_MIN_P       = 4
    # COMMON_SAMPLER_TYPE_TFS_Z     = 5
    COMMON_SAMPLER_TYPE_TYPICAL_P   = 6
    COMMON_SAMPLER_TYPE_TEMPERATURE = 7
    COMMON_SAMPLER_TYPE_XTC         = 8
    COMMON_SAMPLER_TYPE_INFILL      = 9
    COMMON_SAMPLER_TYPE_PENALTIES   = 10

# callbacks
# -----------------------------------------------------------------------------

cdef void log_callback(ggml_log_level level, const char * text, void * py_log_callback) noexcept:
    """ggml_log_callback wrapper to enabling python callbacks to be used"""
    (<object>py_log_callback)(level, text.decode())

def set_log_callback(object py_log_callback):
    """Set callback for all future logging events.

    If this is not called, or NULL is supplied, everything is output on stderr.
    """
    llama_cpp.llama_log_set(<llama_cpp.ggml_log_callback>&log_callback, <void*>py_log_callback)

cdef bint abort_callback(void * py_abort_callback) noexcept:
    """ggml_abort_callback wrapper enabling python callbacks to be used"""
    return (<object>py_abort_callback)()

# cdef cppbool sched_eval_callback(llama_cpp.ggml_tensor * t, cppbool ask, void * py_sched_eval_callback) noexcept:
#     """ggml_backend_sched_eval_callback wrapper enabling python callbacks to be used"""
#     cdef GGMLTensor tensor = GGMLTensor.from_ptr(t)
#     return (<object>py_sched_eval_callback)(tensor, ask)

cdef cppbool progress_callback(float progress, void * py_progress_callback) noexcept:
    """llama_progress_callback callback wrapper enabling python callbacks to be used"""
    return (<object>py_progress_callback)(progress)


# high-level api
# -----------------------------------------------------------------------------


def ask(str prompt, str model, n_predict=512, n_ctx=2048, disable_log=True, n_threads=4) -> str:
    """ask/prompt a llama model"""

    cdef str result = llama_cpp.simple_prompt(
        model.encode(),
        prompt.encode(),
        n_predict,
        n_ctx,
        disable_log,
        n_threads).decode()
    return result.strip()




# wrapper classes
# -----------------------------------------------------------------------------

# alternative implementation with cython memoryviews

# import numpy as np
# cdef class TokenDataArray:
#     """Wrapper for llama_token_data_array using memoryviews for numpy interop."""
#     cdef llama_cpp.llama_token_data_array * ptr
#     cdef bint owner
#     cdef float[:] logits_view
#     cdef int32_t[:] ids_view
#     cdef float[:] probs_view

#     def __cinit__(self):
#         self.ptr = NULL
#         self.owner = True

#     def __dealloc__(self):
#         if self.ptr is not NULL and self.owner is True:
#             if self.ptr.data is not NULL:
#                 free(self.ptr.data)
#             free(self.ptr)
#             self.ptr = NULL

#     def __init__(self, size_t n_vocab):
#         # Allocate the array struct
#         self.ptr = <llama_cpp.llama_token_data_array*>malloc(sizeof(llama_cpp.llama_token_data_array))
#         if self.ptr is NULL:
#             raise MemoryError("Failed to allocate token data array")

#         # Allocate the token data
#         self.ptr.data = <llama_cpp.llama_token_data*>malloc(n_vocab * sizeof(llama_cpp.llama_token_data))
#         if self.ptr.data is NULL:
#             free(self.ptr)
#             self.ptr = NULL
#             raise MemoryError("Failed to allocate token data")

#         self.ptr.size = n_vocab
#         self.ptr.sorted = False

#         # Create memoryviews for numpy interop
#         cdef size_t i
#         for i in range(n_vocab):
#             self.ptr.data[i].id = i
#             self.ptr.data[i].logit = 0
#             self.ptr.data[i].p = 0

#         # Set up memoryviews
#         self.logits_view = <float[:n_vocab]>(<float*>&self.ptr.data[0].logit)  # type: ignore
#         self.ids_view = <int32_t[:n_vocab]>(<int32_t*>&self.ptr.data[0].id)  # type: ignore
#         self.probs_view = <float[:n_vocab]>(<float*>&self.ptr.data[0].p)  # type: ignore

#     @property
#     def logits(self):
#         """Get logits as numpy array view"""
#         return np.asarray(self.logits_view)

#     @property
#     def ids(self):
#         """Get token ids as numpy array view"""
#         return np.asarray(self.ids_view)

#     @property
#     def probabilities(self):
#         """Get probabilities as numpy array view"""
#         return np.asarray(self.probs_view)

#     @property
#     def size(self) -> int:
#         """Get size of array"""
#         return self.ptr.size

#     @property
#     def sorted(self) -> bool:
#         """Check if array is sorted"""
#         return self.ptr.sorted

#     @sorted.setter
#     def sorted(self, value: bool):
#         """Set sorted flag"""
#         self.ptr.sorted = value



cdef class LlamaLogitBias:
    cdef llama_cpp.llama_logit_bias * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, int token, float bias):
        self.ptr = <llama_cpp.llama_logit_bias *>malloc(sizeof(llama_cpp.llama_logit_bias))
        if self.ptr is NULL:
            raise MemoryError
        self.owner = True
        self.ptr.token = token
        self.ptr.bias = bias

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaLogitBias from_ptr(llama_cpp.llama_logit_bias *ptr, bint owner=False):
        cdef LlamaLogitBias wrapper = LlamaLogitBias.__new__(LlamaLogitBias)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @property
    def token(self) -> int:
        """token token"""
        return self.ptr.token

    @token.setter
    def token(self, int value):
        self.ptr.token = value

    @property
    def bias(self) -> float:
        """bias"""
        return self.ptr.bias

    @bias.setter
    def bias(self, float value):
        self.ptr.bias = value


cdef class LlamaTokenData:
    cdef llama_cpp.llama_token_data * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, int id, float logit, float p):
        self.ptr = <llama_cpp.llama_token_data *>malloc(sizeof(llama_cpp.llama_token_data))
        if self.ptr is NULL:
            raise MemoryError
        self.owner = True
        self.ptr.id = id
        self.ptr.logit = logit
        self.ptr.p = p

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaTokenData from_ptr(llama_cpp.llama_token_data *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef LlamaTokenData wrapper = LlamaTokenData.__new__(LlamaTokenData)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @property
    def id(self) -> int:
        """token id"""
        return self.ptr.id

    @id.setter
    def id(self, int value):
        self.ptr.id = value

    @property
    def logit(self) -> float:
        """log-odds of the token"""
        return self.ptr.logit

    @logit.setter
    def logit(self, float value):
        self.ptr.logit = value

    @property
    def p(self) -> float:
        """probability of the token"""
        return self.ptr.p

    @p.setter
    def p(self, float value):
        self.ptr.p = value


cdef class LlamaTokenDataArray:
    """Intermediate Cython wrapper for llama_token_data_array."""
    cdef llama_cpp.llama_token_data_array * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    # def __init__(self, int id, float logit, float p):
    #     self.ptr = <llama_cpp.llama_token_data_array *>malloc(sizeof(llama_cpp.llama_token_data_array))
    #     if self.ptr is NULL:
    #         raise MemoryError
    #     self.owner = True
    #     self.ptr.id = id
    #     self.ptr.logit = logit
    #     self.ptr.p = p

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaTokenDataArray from_ptr(llama_cpp.llama_token_data_array *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef LlamaTokenDataArray wrapper = LlamaTokenDataArray.__new__(LlamaTokenDataArray)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @property
    def data(self) -> list[LlamaTokenData]:
        """llama_token_data array"""
        result = []
        for i in range(self.p.size):
            result.append(LlamaTokenData.from_ptr(<llama_cpp.llama_token_data *>self.p.data[i]))
        return result

    # FIXME: should resize on demand.
    @data.setter
    def data(self, value: list[LlamaTokenData]):
        if len(value) != self.p.size:
            raise RuntimeError("sizes of input array and receiving array should be the samee")
        for i in range(self.p.size):
            self.ptr.data[i].id = value[i].id
            self.ptr.data[i].logit = value[i].logit
            self.ptr.data[i].id = value[i].id

    @property
    def size(self) -> int:
        """size field"""
        return self.ptr.size

    @size.setter
    def size(self, size_t value):
        self.ptr.size = value

    @property
    def selected(self) -> int:
        """this is the index in the data array (i.e. not the token id)"""
        return self.ptr.selected

    @selected.setter
    def selected(self, int64_t value):
        self.ptr.selected = value

    @property
    def sorted(self) -> bool:
        """sorted field"""
        return self.ptr.sorted

    @sorted.setter
    def sorted(self, bint value):
        self.ptr.sorted = value


cdef class LlamaAdapterLora:
    cdef llama_cpp.llama_adapter_lora * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_adapter_lora_free(self.ptr)
            self.ptr = NULL

    def __init__(self):
        # Prevent accidental instantiation from normal Python code
        # since we cannot pass a struct pointer into a Python constructor.
        raise TypeError("This class cannot be instantiated directly.")

    @staticmethod
    cdef LlamaAdapterLora from_ptr(llama_cpp.llama_adapter_lora *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef LlamaAdapterLora wrapper = LlamaAdapterLora.__new__(LlamaAdapterLora)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper


cdef class GgmlBackendDevice:
    cdef llama_cpp.ggml_backend_device * ptr
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
    cdef GgmlBackendDevice from_ptr(llama_cpp.ggml_backend_device *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef GgmlBackendDevice wrapper = GgmlBackendDevice.__new__(GgmlBackendDevice)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper


cdef class GGMLThreadPoolParams:
    # NOTE: should this be a * ptr
    cdef llama_cpp.ggml_threadpool_params p


    def __init__(self, int n_threads):
        self.p = llama_cpp.ggml_threadpool_params_default(n_threads)

    # cdef void ggml_threadpool_params_init(ggml_threadpool_params * p, int n_threads)

    @staticmethod
    cdef GGMLThreadPoolParams from_instance(llama_cpp.ggml_threadpool_params params):
        cdef GGMLThreadPoolParams wrapper = GGMLThreadPoolParams.__new__(GGMLThreadPoolParams)
        wrapper.p = params
        return wrapper

    @staticmethod
    def from_cpu_params(CpuParams params) -> GGMLThreadPoolParams:
        cdef llama_cpp.ggml_threadpool_params tparams = llama_cpp.ggml_threadpool_params_from_cpu_params(params.ptr[0])
        return GGMLThreadPoolParams.from_instance(tparams)

    def match(self, GGMLThreadPoolParams other) -> bool:
        return llama_cpp.ggml_threadpool_params_match(&self.p, &other.p)

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
    def prio(self) -> llama_cpp.ggml_sched_priority:
        """thread priority"""
        return self.p.prio

    @prio.setter
    def prio(self, llama_cpp.ggml_sched_priority value):
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


cdef class GGMLThreadPool:
    cdef llama_cpp.ggml_threadpool * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __init__(self, GGMLThreadPoolParams params):
        self.ptr = llama_cpp.ggml_threadpool_new(&params.p)
        if self.ptr is NULL:
            raise MemoryError
        self.owner = True

    def __dealloc__(self):
        # De-allocate if not null and flag is set
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.ggml_threadpool_free(self.ptr)
            self.ptr = NULL

    # def get_n_threads(self) -> int:
    #     return llama_cpp.ggml_threadpool_get_n_threads(self.ptr)

    def pause(self):
        return llama_cpp.ggml_threadpool_pause(self.ptr)

    def resume(self):
        return llama_cpp.ggml_threadpool_resume(self.ptr)


cdef class GGMLTensor:
    cdef llama_cpp.ggml_tensor * ptr
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
    cdef GGMLTensor from_ptr(llama_cpp.ggml_tensor *ptr, bint owner=False):
        # Fast call to __new__() that bypasses the __init__() constructor.
        cdef GGMLTensor wrapper = GGMLTensor.__new__(GGMLTensor)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @staticmethod
    cdef GGMLTensor create():
        cdef llama_cpp.ggml_tensor *ptr = <llama_cpp.ggml_tensor *>malloc(sizeof(llama_cpp.ggml_tensor))
        if ptr is NULL:
            raise MemoryError
        # ptr.a = 0
        # ptr.b = 0
        return GGMLTensor.from_ptr(ptr, owner=True)


cdef class LlamaSamplerChainParams:
    cdef llama_cpp.llama_sampler_chain_params p

    def __init__(self):
        self.p = llama_cpp.llama_sampler_chain_default_params()

    @staticmethod
    cdef LlamaSamplerChainParams from_instance(llama_cpp.llama_sampler_chain_params params):
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


cdef class LlamaSampler:
    """cython wrapper for llama_cpp.llama_sampler."""
    cdef llama_cpp.llama_sampler * ptr
    cdef LlamaSamplerChainParams params
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, params: Optional[LlamaSamplerChainParams] = None):
        if not params:
            self.ptr = llama_cpp.llama_sampler_chain_init(self.params.p)
        else:
            self.ptr = llama_cpp.llama_sampler_chain_init(params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to init Sampler")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_sampler_free(self.ptr)
            self.ptr = NULL

    def name(self) -> str:
        """Get sampler name"""
        return llama_cpp.llama_sampler_name(self.ptr).decode()

    def accept(self, llama_cpp.llama_token token):
        """Accept llama token"""
        llama_cpp.llama_sampler_accept(self.ptr, token)

    # cdef void llama_sampler_apply (llama_sampler * smpl, llama_token_data_array * cur_p)

    def reset(self):
        """Reset sampler"""
        llama_cpp.llama_sampler_reset(self.ptr)

    def clone(self) -> LlamaSampler:
        """clone sampler"""
        cdef llama_cpp.llama_sampler * smplr = llama_cpp.llama_sampler_clone(self.ptr)
        cdef LlamaSampler wrapper = LlamaSampler.__new__(LlamaSampler)
        wrapper.ptr = smplr
        return wrapper

    def get_seed(self) -> int:
        """Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise"""
        return llama_cpp.llama_sampler_get_seed(self.ptr)

    def add_greedy(self):
        """Add greedy sampling chain link

        This should be at the end of the chain.
        """
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_greedy())

    def add_dist(self, uint32_t seed):
        """Add dist sampling chain link

        This should be at the end of the chain.
        """
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_dist(seed))

    # DEPRECATED
    # def add_softmax(self):
    #     """Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits."""
    #     llama_cpp.llama_sampler_chain_add(
    #         self.ptr, llama_cpp.llama_sampler_init_softmax())

    def add_top_k(self, int32_t k):
        """Add Top-K sampling chain link.

        Described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_top_k(k))

    def add_top_p(self, float p, size_t min_keep):
        """Add Nucleus sampling chain link.

        Described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_top_p(p, min_keep))

    def add_min_p(self, float p, size_t min_keep):
        """Add Minimum P sampling.

        Described in https:#github.com/ggerganov/llama.cpp/pull/3841"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_min_p(p, min_keep))

    def add_typical(self, float p, size_t min_keep):
        """Add Locally Typical Sampling implementation.

        Described in the paper https:#arxiv.org/abs/2202.00666."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_typical(p, min_keep))

    def add_temp(self, float t):
        """Add temperature sampling chain link.

        Updates the logits `l_i = l_i/t`. When `t <= 0.0f`,
        the maximum logit is kept at its original value, the rest are set to -inf."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_temp(t))

    def add_temp_ext(self, float t, float delta, float exponent):
        """Add Dynamic temperature implementation sampling chain link

        Described in the paper https:#arxiv.org/abs/2309.02772."""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_temp_ext(t, delta, exponent))

    def add_xtc(self, float p, float t, size_t min_keep, uint32_t seed):
        """Add XTC sampler chain link

        Described in https://github.com/oobabooga/text-generation-webui/pull/6335"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_xtc(p, t, min_keep, seed))

    # XXX: docstring incorrect
    def add_mirostat(self, int n_vocab, uint32_t seed, float tau, float eta, int m):
        """Mirostat 1.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.

        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau:     The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta:     The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        m:       The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
        mu:      Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
        """
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m))

    def add_mirostat_v2(self, uint32_t seed, float tau, float eta):
        """Mirostat 2.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.

        candidates: A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
        tau:  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
        eta: The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
        mu: Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
        """
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_mirostat_v2(seed, tau, eta))

    def add_grammar(self, LlamaVocab vocab, str grammar_str, str grammar_root):
        """Add grammer chain link"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_grammar(
                vocab.ptr, grammar_str.encode(), grammar_root.encode()))

    def add_penalties(self,
         int penalty_last_n,   # last n tokens to penalize (0 = disable penalty, -1 = context size)
       float penalty_repeat,   # 1.0 = disabled
       float penalty_freq,     # 0.0 = disabled
       float penalty_present): # 0.0 = disabled
        """Add penalties chain link"""
        llama_cpp.llama_sampler_chain_add(
            self.ptr, llama_cpp.llama_sampler_init_penalties(
                penalty_last_n,
                penalty_repeat,
                penalty_freq,
                penalty_present,
            ))

    # XXX FIXME:
    # def add_logit_bias(self, int n_vocab, int n_logit_bias, logit_bias: list[LogitBias]):
    #     """Add grammer chain link"""
    #     cdef vector[llama_cpp.logit_bias] vec
    #     llama_cpp.llama_sampler_chain_add(
    #         self.ptr, llama_cpp.llama_sampler_init_logit_bias(
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
        llama_cpp.llama_sampler_chain_add(self.ptr,
            llama_cpp.llama_sampler_init_infill(vocab.ptr))


    def sample(self, LlamaContext ctx, int idx) -> int:
        """Sample and accept a token from the idx-th output of the last evaluation

        Shorthand for:

           const auto * logits = llama_get_logits_ith(ctx, idx)
           llama_token_data_array cur_p = { ... init from logits ... }
           llama_sampler_apply(smpl, &cur_p)
           return cur_p.data[cur_p.selected].id

        At this point, this is mostly a convenience function.
        """
        return llama_cpp.llama_sampler_sample(self.ptr, ctx.ptr, idx)


    #
    # Performance utils
    #
    # NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.
    #


    # def get_perf_data(self):
    #     """
    #     # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    #     """
    #     cdef llama_perf_sampler_data data = llama_cpp.llama_perf_sampler(self.ptr)

    def print_perf_data(self):
        """
        # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
        """
        llama_cpp.llama_perf_sampler_print(self.ptr)

    def reset_perf_data(self):
        """
        # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
        """
        llama_cpp.llama_perf_sampler_reset(self.ptr)



cdef class LlamaChatMessage:
    """cython wrapper for llama_cpp.llama_chat_message

    members role and content are const char *
    """
    cdef llama_cpp.llama_chat_message p

    # def __cinit__(self, role: str, content: str):
    #     cdef int role_len = len(role) + 1
    #     cdef int content_len = len(content) + 1
    #     cdef char* role_ptr = <char*>malloc(sizeof(char)*role_len)
    #     cdef char* content_ptr = <char*>malloc(sizeof(char)*content_len)
    #     strcpy(role_ptr, role.encode())
    #     strcpy(content_ptr, content.encode())
    #     self.p.role = role_ptr
    #     self.p.content = content_ptr

    @staticmethod
    cdef LlamaChatMessage from_instance(llama_cpp.llama_chat_message msg):
        cdef LlamaChatMessage wrapper = LlamaChatMessage.__new__(LlamaChatMessage)
        wrapper.p = msg
        return wrapper

    @property
    def role(self) -> str:
        """readonly chat role"""
        return self.p.role.decode()

    @property
    def content(self) -> str:
        """readonly chat content"""
        return self.p.content.decode()



# cdef class CommonChatMsg:
#     """cython wrapper for llama_cpp.common_chat_msg"""
#     cdef llama_cpp.common_chat_msg p

#     def __init__(self, str role, str content):
#         self.p.role = role.encode()
#         self.p.content = content.encode()

#     @staticmethod
#     cdef CommonChatMsg from_instance(llama_cpp.common_chat_msg msg):
#         cdef CommonChatMsg wrapper = CommonChatMsg.__new__(CommonChatMsg)
#         wrapper.p = msg
#         return wrapper

#     @property
#     def role(self) -> str:
#         return self.p.role.decode()

#     @role.setter
#     def role(self, str value):
#         self.p.role = value.encode()

#     @property
#     def content(self) -> str:
#         return self.p.content.decode()

#     @content.setter
#     def content(self, str value):
#         self.p.content = value.encode()



cdef class CommonParamsSampling:
    cdef llama_cpp.common_params_sampling p

    @staticmethod
    cdef CommonParamsSampling from_instance(llama_cpp.common_params_sampling params):
        cdef CommonParamsSampling wrapper = CommonParamsSampling.__new__(CommonParamsSampling)
        wrapper.p = params
        return wrapper

    def print(self) -> str:
        """print the parameters into a string"""
        return (
            "\trepeat_last_n = %d, repeat_penalty = %.3f, frequency_penalty = %.3f, presence_penalty = %.3f\n"
            "\tdry_multiplier = %.3f, dry_base = %.3f, dry_allowed_length = %d, dry_penalty_last_n = %d\n"
            "\ttop_k = %d, top_p = %.3f, min_p = %.3f, xtc_probability = %.3f, xtc_threshold = %.3f, typical_p = %.3f, temp = %.3f\n"
            "\tmirostat = %d, mirostat_lr = %.3f, mirostat_ent = %.3f" % (
                self.penalty_last_n, self.penalty_repeat, self.penalty_freq, self.penalty_present,
                self.dry_multiplier, self.dry_base, self.dry_allowed_length, self.dry_penalty_last_n,
                self.top_k, self.top_p, self.min_p, self.xtc_probability, self.xtc_threshold, self.typ_p, self.temp,
                self.mirostat, self.mirostat_eta, self.mirostat_tau)
        )

    @property
    def seed(self) -> int:
        """the seed used to initialize llama_sampler."""
        return self.p.seed

    @seed.setter
    def seed(self, uint32_t value):
        self.p.seed = value

    @property
    def n_prev(self) -> int:
        """number of previous tokens to remember"""
        return self.p.n_prev

    @n_prev.setter
    def n_prev(self, int32_t value):
        self.p.n_prev = value

    @property
    def n_probs(self) -> int:
        """if greater than 0, output the probabilities of top n_probs tokens."""
        return self.p.n_probs

    @n_probs.setter
    def n_probs(self, int32_t value):
        self.p.n_probs = value

    @property
    def min_keep(self) -> int:
        """if greater than 0, output the probabilities of top min_keep tokens."""
        return self.p.min_keep

    @min_keep.setter
    def min_keep(self, int32_t value):
        self.p.min_keep = value

    @property
    def top_k(self) -> int:
        """<= 0 to use vocab size."""
        return self.p.top_k

    @top_k.setter
    def top_k(self, int32_t value):
        self.p.top_k = value

    @property
    def top_p(self) -> float:
        """1.0 = disabled"""
        return self.p.top_p

    @top_p.setter
    def top_p(self, float value):
        self.p.top_p = value

    @property
    def min_p(self) -> float:
        """0.0 = disabled"""
        return self.p.min_p

    @min_p.setter
    def min_p(self, float value):
        self.p.min_p = value

    @property
    def xtc_probability(self) -> float:
        """0.0 = disabled"""
        return self.p.xtc_probability

    @xtc_probability.setter
    def xtc_probability(self, float value):
        self.p.xtc_probability = value

    @property
    def xtc_threshold(self) -> float:
        """> 0.5 disables XTC"""
        return self.p.xtc_threshold

    @xtc_threshold.setter
    def xtc_threshold(self, float value):
        self.p.xtc_threshold = value

    # @property
    # def tfs_z(self) -> float:
    #     """1.0 = disabled"""
    #     return self.p.tfs_z

    # @tfs_z.setter
    # def tfs_z(self, float value):
    #     self.p.tfs_z = value

    @property
    def typ_p(self) -> float:
        """typical_p, 1.0 = disabled"""
        return self.p.typ_p

    @typ_p.setter
    def typ_p(self, float value):
        self.p.typ_p = value

    @property
    def temp(self) -> float:
        """<= 0.0 to sample greedily, 0.0 to not output probabilities"""
        return self.p.temp

    @temp.setter
    def temp(self, float value):
        self.p.temp = value

    @property
    def dynatemp_range(self) -> float:
        """0.0 = disabled"""
        return self.p.dynatemp_range

    @dynatemp_range.setter
    def dynatemp_range(self, float value):
        self.p.dynatemp_range = value

    @property
    def dynatemp_exponent(self) -> float:
        """controls how entropy maps to temperature in dynamic temperature sampler"""
        return self.p.dynatemp_exponent

    @dynatemp_exponent.setter
    def dynatemp_exponent(self, float value):
        self.p.dynatemp_exponent = value

    @property
    def penalty_last_n(self) -> int:
        """last n tokens to penalize (0 = disable penalty, -1 = context size)"""
        return self.p.penalty_last_n

    @penalty_last_n.setter
    def penalty_last_n(self, int value):
        self.p.penalty_last_n = value

    @property
    def penalty_repeat(self) -> float:
        """1.0 = disabled"""
        return self.p.penalty_repeat

    @penalty_repeat.setter
    def penalty_repeat(self, float value):
        self.p.penalty_repeat = value

    @property
    def penalty_freq(self) -> float:
        """0.0 = disabled"""
        return self.p.penalty_freq

    @penalty_freq.setter
    def penalty_freq(self, float value):
        self.p.penalty_freq = value

    @property
    def penalty_present(self) -> float:
        """0.0 = disabled"""
        return self.p.penalty_present

    @penalty_present.setter
    def penalty_present(self, float value):
        self.p.penalty_present = value

    @property
    def dry_multiplier(self) -> float:
        """0.0 = disabled

        DRY repetition penalty for tokens extending repetition
        """
        return self.p.dry_multiplier

    @dry_multiplier.setter
    def dry_multiplier(self, float value):
        self.p.dry_multiplier = value

    @property
    def dry_base(self) -> float:
        """0.0 = disabled

        multiplier * base ^ (length of sequence before token - allowed length)
        """
        return self.p.dry_base

    @dry_base.setter
    def dry_base(self, float value):
        self.p.dry_base = value

    @property
    def dry_allowed_length(self) -> int:
        """tokens extending repetitions beyond this receive penalty"""
        return self.p.dry_allowed_length

    @dry_allowed_length.setter
    def dry_allowed_length(self, int value):
        self.p.dry_allowed_length = value

    @property
    def dry_penalty_last_n(self) -> int:
        """how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)"""
        return self.p.dry_penalty_last_n

    @dry_penalty_last_n.setter
    def dry_penalty_last_n(self, int value):
        self.p.dry_penalty_last_n = value

    @property
    def mirostat(self) -> int:
        """0 = disabled, 1 = mirostat, 2 = mirostat 2.0"""
        return self.p.mirostat

    @mirostat.setter
    def mirostat(self, int value):
        self.p.mirostat = value

    @property
    def mirostat_tau(self) -> float:
        """target entropy"""
        return self.p.mirostat_tau

    @mirostat_tau.setter
    def mirostat_tau(self, float value):
        self.p.mirostat_tau = value

    @property
    def mirostat_eta(self) -> float:
        """learning rate"""
        return self.p.mirostat_eta

    @mirostat_eta.setter
    def mirostat_eta(self, float value):
        self.p.mirostat_eta = value

    # @property
    # def penalize_nl(self) -> bool:
    #     """consider newlines as a repeatable token"""
    #     return self.p.penalize_nl

    # @penalize_nl.setter
    # def penalize_nl(self, bint value):
    #     self.p.penalize_nl = value

    @property
    def ignore_eos(self) -> bool:
        """ignore end-of-sentence"""
        return self.p.ignore_eos

    @ignore_eos.setter
    def ignore_eos(self, bint value):
        self.p.ignore_eos = value

    @property
    def no_perf(self) -> bool:
        """disable performance metrics"""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, bint value):
        self.p.no_perf = value

    @property
    def samplers(self) -> list[common_sampler_type]:
        """get/set sampler types

        std_vector[common_sampler_type] samplers
        """
        return self.p.samplers

    @samplers.setter
    def samplers(self, value: list[common_sampler_type]):
        self.p.samplers = value

    @property
    def grammar(self) -> str:
        """optional BNF-like grammar to constrain sampling"""
        return self.p.grammar

    @grammar.setter
    def grammar(self, str value):
        self.p.grammar = value

    @property
    def logit_bias(self) -> list[LlamaLogitBias]:
        """logit biases to apply

        std_vector[llama_logit_bias] logit_bias
        """
        result = []
        for i in range(self.p.logit_bias.size()):
            result.append(LlamaLogitBias.from_ptr(&self.p.logit_bias[i]))
        return result

    @logit_bias.setter
    def logit_bias(self, elems: list[LlamaLogitBias]):
        cdef vector[llama_cpp.llama_logit_bias] vec
        for elem in elems:
            vec.push_back(elem.ptr[0])
        self.p.logit_bias = vec




cdef class CommonSampler:
    """cython wrapper of llama_cpp.common_sampler"""
    cdef llama_cpp.common_sampler * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, LlamaModel model, CommonParamsSampling params):
        self.ptr = llama_cpp.common_sampler_init(model.ptr, params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to init Sampler")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.common_sampler_free(self.ptr)
            self.ptr = NULL

    def accept(self, llama_cpp.llama_token token, bint accept_grammar):
        """if accept_grammar is true, the token is accepted both by the sampling chain and the grammar"""
        llama_cpp.common_sampler_accept(self.ptr, token, accept_grammar)

    def reset(self):
        """reset common sampler"""
        llama_cpp.common_sampler_reset(self.ptr)

    def clone(self) -> CommonSampler:
        """clone sampler"""
        cdef llama_cpp.common_sampler * smplr = llama_cpp.common_sampler_clone(self.ptr)
        cdef CommonSampler wrapper = CommonSampler.__new__(CommonSampler)
        wrapper.ptr = smplr
        return wrapper

    def sample(self, LlamaContext ctx, int idx, bint grammar_first) -> int:
        """if grammar_first is true, the grammar is applied before the samplers (slower)

        useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
        """
        return llama_cpp.common_sampler_sample(self.ptr, ctx.ptr, idx, grammar_first)


    # generalized version of common_sampler_sample
    #
    # will cross-reference the sampled tokens with a batch of draft tokens and accept those that match
    # if the sampler disagrees at some point, we stop and return the accepted tokens up to now
    #
    #      common_sampler_sample_n(gsmpl, ctx, { idx }, {});
    #
    # is equivalent to
    #
    #      common_sampler_sample(gsmpl, ctx, idx);
    #      common_sampler_accept(gsmpl, token, true);
    #
    # requires: idxs.size() == draft.size() + 1
    #
    # returns at least 1 token, up to idxs.size()

    # std_vector[llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama_context * ctx, const std_vector[int] & idxs, const llama_tokens & draft, bint grammar_first)

    # assume idxs == [ 0, 1, 2, ..., draft.size() ]
    # std_vector[llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama_context * ctx, const llama_tokens & draft, bint grammar_first)


    def get_seed(self) -> int:
        """get random seed"""
        return llama_cpp.common_sampler_get_seed(self.ptr)

    # def get_candidates(self) -> LlamaTokenDataArray:
    #     """access the internal list of current candidate tokens"""
    #     return llama_cpp.common_sampler_get_candidates(self.ptr)

    def get_last(self) -> int:
        """get the last accepted token"""
        return llama_cpp.common_sampler_last(self.ptr)

    def print(self) -> str:
        """print the sampler chain into a string"""
        return llama_cpp.common_sampler_print(self.ptr).decode()

    def prev_str(self, LlamaContext ctx, int n) -> str:
        """get a string representation of the last accepted tokens"""
        return llama_cpp.common_sampler_prev_str(self.ptr, ctx.ptr, n).decode()

    # char common_sampler_type_to_chr(common_sampler_type cnstr)
    # std_string common_sampler_type_to_str(common_sampler_type cnstr)

    # std_vector[common_sampler_type] common_sampler_types_from_names(const std_vector[std_string] & names, bint allow_alt_names)
    # std_vector[common_sampler_type] common_sampler_types_from_chars(const std_string & chars)


cdef class CpuParams:
    cdef llama_cpp.cpu_params *ptr
    cdef object owner

    @staticmethod
    cdef CpuParams from_ptr(llama_cpp.cpu_params *p, object parent):
        cdef CpuParams wrapper = CpuParams.__new__(CpuParams)
        wrapper.ptr = p
        wrapper.owner = parent
        return wrapper

    @property
    def n_threads(self) -> int:
        """number of threads."""
        return self.ptr.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self.ptr.n_threads = value

    @property
    def cpumask(self) -> list[bool]:
        """CPU affinity mask: mask of cpu cores (all-zeros means use default affinity settings)

        cpumask[GGML_MAX_N_THREADS] is (by default) of size 16
        """
        res = []
        for i in range(GGML_MAX_N_THREADS):
            res.append(<bint>self.ptr.cpumask[i])
        return res

    @cpumask.setter
    def cpumask(self, values: list[bool]):
        assert len(values) == GGML_MAX_N_THREADS
        for i in range(GGML_MAX_N_THREADS):
            self.ptr.cpumask[i] = <bint>values[i]

    @property
    def mask_valid(self) -> bool:
        """Default: any CPU."""
        return self.ptr.mask_valid

    @mask_valid.setter
    def mask_valid(self, value: bool):
        self.ptr.mask_valid = value

    @property
    def priority(self) -> llama_cpp.ggml_sched_priority:
        """Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)."""
        return self.ptr.priority

    @priority.setter
    def priority(self, value: llama_cpp.ggml_sched_priority):
        self.ptr.priority = value

    @property
    def strict_cpu(self) -> bool:
        """Use strict CPU placement."""
        return self.ptr.strict_cpu

    @strict_cpu.setter
    def strict_cpu(self, bint value):
        self.ptr.strict_cpu = value

    @property
    def poll(self) -> uint32_t:
        """Polling (busywait) level (0 - no polling, 100 - mostly polling)"""
        return self.ptr.poll

    @poll.setter
    def poll(self, uint32_t value):
        self.ptr.poll = value


cdef class CommonParamsSpeculative:
    cdef llama_cpp.common_params_speculative p
    # cdef public CpuParams cpuparams
    # cdef public CpuParams cpuparams_batch
    # cdef vector[GgmlBackendDevice] devices

    @staticmethod
    cdef CommonParamsSpeculative from_instance(llama_cpp.common_params_speculative params):
        cdef CommonParamsSpeculative wrapper = CommonParamsSpeculative.__new__(CommonParamsSpeculative)
        wrapper.p = params
        # wrapper.cpuparams = CpuParams.from_ptr(&wrapper.p.cpuparams, wrapper)
        # wrapper.cpuparams_batch = CpuParams.from_ptr(&wrapper.p.cpuparams_batch, wrapper)
        return wrapper

    @property
    def n_ctx(self) -> int:
        """draft context size."""
        return self.p.n_ctx

    @n_ctx.setter
    def n_ctx(self, value: int):
        self.p.n_ctx = value

    @property
    def n_max(self) -> int:
        """maximum number of tokens to draft during speculative decoding."""
        return self.p.n_max

    @n_max.setter
    def n_max(self, value: int):
        self.p.n_max = value

    @property
    def n_min(self) -> int:
        """minimum number of draft tokens to use for speculative decoding."""
        return self.p.n_min

    @n_min.setter
    def n_min(self, value: int):
        self.p.n_min = value

    @property
    def n_gpu_layers(self) -> int:
        """number of layers to store in VRAM (-1 - use default)."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def p_split(self) -> float:
        """speculative decoding split probability."""
        return self.p.p_split

    @p_split.setter
    def p_split(self, value: float):
        self.p.p_split = value

    @property
    def p_min(self) -> float:
        """minimum speculative decoding probability (greedy)."""
        return self.p.p_min

    @p_min.setter
    def p_min(self, value: float):
        self.p.p_min = value

    @property
    def model(self) -> str:
        """ draft model for speculative decoding."""
        return self.p.model

    @model.setter
    def model(self, value: str):
        self.p.model = value


cdef class CommonParamsVocoder:
    cdef llama_cpp.common_params_vocoder p

    def __init__(self, hf_repo: str = "", hf_file: str = "", model: str = "", model_url: str = ""):
        self.hf_repo = hf_repo
        self.hf_file = hf_file
        self.model = model
        self.model_url = model_url

    @staticmethod
    cdef CommonParamsVocoder from_instance(llama_cpp.common_params_vocoder params):
        cdef CommonParamsVocoder wrapper = CommonParamsVocoder.__new__(CommonParamsVocoder)
        wrapper.p = params
        return wrapper

    @property
    def hf_repo(self) -> str:
        """HF repo"""
        return self.p.hf_repo.decode()

    @hf_repo.setter
    def hf_repo(self, value: str):
        self.p.hf_repo = value.encode()

    @property
    def hf_file(self) -> str:
        """HF file"""
        return self.p.hf_file.decode()

    @hf_file.setter
    def hf_file(self, value: str):
        self.p.hf_file = value.encode()

    @property
    def model(self) -> str:
        """model path"""
        return self.p.model.decode()

    @model.setter
    def model(self, value: str):
        self.p.model = value.encode()

    # @property
    # def model_url(self) -> str:
    #     """model url to download."""
    #     return self.p.model_url.decode()

    # @model_url.setter
    # def model_url(self, value: str):
    #     self.p.model_url = value.encode()


# cdef class CommonParamsModel:
#     cdef llama_cpp.common_params_model m

#     @property
#     def path(self) -> str:
#         """model path"""
#         return self.p.model.decode()

#     @path.setter
#     def path(self, value: str):
#         self.m.path = value.encode()


cdef class CommonParams:
    cdef llama_cpp.common_params p
    cdef public CpuParams cpuparams
    cdef public CpuParams cpuparams_batch
    # cdef public CpuParams draft_cpuparams
    # cdef public CpuParams draft_cpuparams_batch
    cdef list[GgmlBackendDevice] devices

    @staticmethod
    cdef CommonParams from_instance(llama_cpp.common_params p):
        cdef CommonParams wrapper = CommonParams.__new__(CommonParams)
        wrapper.p = p
        wrapper.cpuparams = CpuParams.from_ptr(&wrapper.p.cpuparams, wrapper)
        wrapper.cpuparams_batch = CpuParams.from_ptr(&wrapper.p.cpuparams_batch, wrapper)
        # wrapper.draft_cpuparams = CpuParams.from_ptr(&wrapper.p.draft_cpuparams, wrapper)
        # wrapper.draft_cpuparams_batch = CpuParams.from_ptr(&wrapper.p.draft_cpuparams_batch, wrapper)
        return wrapper

    def __cinit__(self):
        # self.p.cb_eval = &sched_eval_callback # set callback wrapper
        self.cpuparams = CpuParams.from_ptr(&self.p.cpuparams, self)
        self.cpuparams_batch = CpuParams.from_ptr(&self.p.cpuparams_batch, self)
        # self.draft_cpuparams = CpuParams.from_ptr(&self.p.draft_cpuparams, self)
        # self.draft_cpuparams_batch = CpuParams.from_ptr(&self.p.draft_cpuparams_batch, self)

    @property
    def n_predict(self) -> int:
        """new tokens to predict."""
        return self.p.n_predict

    @n_predict.setter
    def n_predict(self, value: int):
        self.p.n_predict = value

    @property
    def n_ctx(self) -> int:
        """context size."""
        return self.p.n_ctx

    @n_ctx.setter
    def n_ctx(self, value: int):
        self.p.n_ctx = value

    @property
    def n_batch(self) -> int:
        """logical batch size for prompt processing (must be >=32)."""
        return self.p.n_batch

    @n_batch.setter
    def n_batch(self, value: int):
        self.p.n_batch = value

    @property
    def n_ubatch(self) -> int:
        """physical batch size for prompt processing (must be >=32)."""
        return self.p.n_ubatch

    @n_ubatch.setter
    def n_ubatch(self, value: int):
        self.p.n_ubatch = value

    @property
    def n_keep(self) -> int:
        """number of tokens to keep from initial prompt."""
        return self.p.n_keep

    @n_keep.setter
    def n_keep(self, value: int):
        self.p.n_keep = value

    @property
    def n_chunks(self) -> int:
        """max number of chunks to process (-1 = unlimited)."""
        return self.p.n_chunks

    @n_chunks.setter
    def n_chunks(self, value: int):
        self.p.n_chunks = value

    @property
    def n_parallel(self) -> int:
        """number of parallel sequences to decode."""
        return self.p.n_parallel

    @n_parallel.setter
    def n_parallel(self, value: int):
        self.p.n_parallel = value

    @property
    def n_sequences(self) -> int:
        """number of sequences to decode."""
        return self.p.n_sequences

    @n_sequences.setter
    def n_sequences(self, value: int):
        self.p.n_sequences = value

    @property
    def grp_attn_n(self) -> int:
        """group-attention factor."""
        return self.p.grp_attn_n

    @grp_attn_n.setter
    def grp_attn_n(self, value: int):
        self.p.grp_attn_n = value

    @property
    def grp_attn_w(self) -> int:
        """group-attention width."""
        return self.p.grp_attn_w

    @grp_attn_w.setter
    def grp_attn_w(self, value: int):
        self.p.grp_attn_w = value

    @property
    def n_print(self) -> int:
        """print token count every n tokens (-1 = disabled)."""
        return self.p.n_print

    @n_print.setter
    def n_print(self, value: int):
        self.p.n_print = value

    @property
    def rope_freq_base(self) -> float:
        """RoPE base frequency."""
        return self.p.rope_freq_base

    @rope_freq_base.setter
    def rope_freq_base(self, value: float):
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

    @property
    def defrag_thold(self) -> float:
        """KV cache defragmentation threshold."""
        return self.p.defrag_thold

    @defrag_thold.setter
    def defrag_thold(self, value: float):
        self.p.defrag_thold = value

    @property
    def n_gpu_layers(self) -> int:
        """number of layers to store in VRAM (-1 - use default)."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def main_gpu(self) -> int:
        """he GPU that is used for scratch and small tensors"""
        return self.p.main_gpu

    @main_gpu.setter
    def main_gpu(self, value: int):
        self.p.main_gpu = value

    @property
    def tensor_split(self) -> list[float]:
        """how split tensors should be distributed across GPUs."""
        result = []
        for i in range(128):
            result.append(self.p.tensor_split[i])
        return result

    @tensor_split.setter
    def tensor_split(self, value: list[float]):
        assert len(value) == 128, "tensor must of length 128"
        for i in range(128):
            self.p.tensor_split[i] = value[i]

    @property
    def split_mode(self) -> llama_split_mode:
        """how to split the model across GPUs."""
        return self.p.split_mode

    @split_mode.setter
    def split_mode(self, llama_split_mode value):
        self.p.split_mode = value

    # @property
    # def cb_eval(self) -> py_sched_eval_callback:
    #     """get/set python ggml backend sched eval callback."""
    #     return <object>self.p.cb_eval_user_data

    # @cb_eval.setter
    # def cb_eval(self, object py_sched_eval_callback):
    #     self.p.cb_eval_user_data = <void*>py_sched_eval_callback

    @property
    def numa(self) -> ggml_numa_strategy:
        """KV cache defragmentation threshold."""
        return self.p.numa

    @numa.setter
    def numa(self, value: ggml_numa_strategy):
        self.p.numa = value

    @property
    def rope_scaling_type(self) -> llama_rope_scaling_type:
        """rope scaling type."""
        return llama_rope_scaling_type(self.p.rope_scaling_type)

    @rope_scaling_type.setter
    def rope_scaling_type(self, llama_rope_scaling_type value):
        self.p.rope_scaling_type = value

    @property
    def pooling_type(self) -> llama_pooling_type:
        """pooling type for embeddings."""
        return (self.p.pooling_type)

    @pooling_type.setter
    def pooling_type(self, llama_pooling_type value):
        self.p.pooling_type = value

    @property
    def attention_type(self) -> llama_attention_type:
        """attention type for embeddings."""
        return llama_attention_type(self.p.attention_type)

    @attention_type.setter
    def attention_type(self, llama_attention_type value):
        self.p.attention_type = value

    @property
    def sampling(self) -> CommonParamsSampling:
        """common params sampling."""
        return CommonParamsSampling.from_instance(self.p.sampling)

    @sampling.setter
    def sampling(self, value: CommonParamsSampling):
        self.p.sampling = value.p

    @property
    def speculative(self) -> CommonParamsSpeculative:
        """common params speculative."""
        return CommonParamsSpeculative.from_instance(self.p.speculative)

    @speculative.setter
    def speculative(self, value: CommonParamsSpeculative):
        self.p.speculative = value.p

    @property
    def vocoder(self) -> CommonParamsVocoder:
        """common params vocoder."""
        return CommonParamsVocoder.from_instance(self.p.vocoder)

    @vocoder.setter
    def vocoder(self, value: CommonParamsVocoder):
        self.p.vocoder = value.p

    @property
    def model(self) -> str:
        """model path"""
        return self.p.model.path.decode()

    @model.setter
    def model(self, value: str):
        self.p.model.path = value.encode('utf8')

    @property
    def model_alias(self) -> str:
        """model alias"""
        return self.p.model_alias.decode()

    @model_alias.setter
    def model_alias(self, value: str):
        self.p.model_alias = value.encode('utf8')

    # @property
    # def model_url(self) -> str:
    #     """model url to download """
    #     return self.p.model_url.decode()

    # @model_url.setter
    # def model_url(self, value: str):
    #     self.p.model_url = value.encode('utf8')

    @property
    def hf_token(self) -> str:
        """hf token"""
        return self.p.hf_token.decode()

    @hf_token.setter
    def hf_token(self, value: str):
        self.p.hf_token = value.encode('utf8')

    # @property
    # def hf_repo(self) -> str:
    #     """hf repo"""
    #     return self.p.hf_repo.decode()

    # @hf_repo.setter
    # def hf_repo(self, value: str):
    #     self.p.hf_repo = value.encode('utf8')

    # @property
    # def hf_file(self) -> str:
    #     """hf file"""
    #     return self.p.hf_file.decode()

    # @hf_file.setter
    # def hf_file(self, value: str):
    #     self.p.hf_file = value.encode('utf8')

    @property
    def system_prompt(self) -> str:
        """the system_prompt text"""
        return self.p.system_prompt.decode()

    @system_prompt.setter
    def system_prompt(self, value: str):
        self.p.system_prompt = value.encode('utf8')


    @property
    def prompt(self) -> str:
        """the prompt text"""
        return self.p.prompt.decode()

    @prompt.setter
    def prompt(self, value: str):
        self.p.prompt = value.encode('utf8')

    @property
    def prompt_file(self) -> str:
        """store the external prompt file name"""
        return self.p.prompt_file.decode()

    @prompt_file.setter
    def prompt_file(self, value: str):
        self.p.prompt_file = value.encode('utf8')

    @property
    def path_prompt_cache(self) -> str:
        """path to file for saving/loading prompt eval state"""
        return self.p.path_prompt_cache.decode()

    @path_prompt_cache.setter
    def path_prompt_cache(self, value: str):
        self.p.path_prompt_cache = value.encode('utf8')

    @property
    def input_prefix(self) -> str:
        """string to prefix user inputs with"""
        return self.p.input_prefix.decode()

    @input_prefix.setter
    def input_prefix(self, value: str):
        self.p.input_prefix = value.encode('utf8')

    @property
    def input_suffix(self) -> str:
        """string to suffix user inputs with"""
        return self.p.input_suffix.decode()

    @input_suffix.setter
    def input_suffix(self, value: str):
        self.p.input_suffix = value.encode('utf8')

    @property
    def lookup_cache_static(self) -> str:
        """path of static ngram cache file for lookup decoding"""
        return self.p.lookup_cache_static.decode()

    @lookup_cache_static.setter
    def lookup_cache_static(self, value: str):
        self.p.lookup_cache_static = value.encode('utf8')

    @property
    def lookup_cache_dynamic(self) -> str:
        """path of dynamic ngram cache file for lookup decoding"""
        return self.p.lookup_cache_dynamic.decode()

    @lookup_cache_dynamic.setter
    def lookup_cache_dynamic(self, value: str):
        self.p.lookup_cache_dynamic = value.encode('utf8')

    @property
    def logits_file(self) -> str:
        """file for saving *all* logits"""
        return self.p.logits_file.decode()

    @logits_file.setter
    def logits_file(self, value: str):
        self.p.logits_file = value.encode('utf8')

    @property
    def in_files(self) -> list[str]:
        """all input files."""
        result = []
        for i in range(self.p.in_files.size()):
            result.append(self.p.in_files[i].decode())
        return result

    @in_files.setter
    def in_files(self, files: list[str]):
        self.p.in_files.clear()
        for i in files:
            self.p.in_files.push_back(i.encode('utf8'))

    @property
    def antiprompt(self) -> list[str]:
        """strings upon which more user input is prompted (a.k.a. reverse prompts)."""
        result = []
        for i in range(self.p.antiprompt.size()):
            result.append(self.p.antiprompt[i].decode())
        return result

    @antiprompt.setter
    def antiprompt(self, values: list[str]):
        self.p.antiprompt.clear()
        for i in values:
            self.p.antiprompt.push_back(i.encode('utf8'))

    # std::vector<llama_model_kv_override> kv_overrides;

    @property
    def lora_init_without_apply(self) -> bool:
        """only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_adapter_lora_apply)."""
        return self.p.lora_init_without_apply

    @lora_init_without_apply.setter
    def lora_init_without_apply(self, value: bool):
        self.p.lora_init_without_apply = value

    # std::vector<llama_adapter_lora_info> lora_adapters; // lora adapter path with user defined scale

    # std::vector<llama_control_vector_load_info> control_vectors; // control vector with user defined scale


    @property
    def verbosity(self) -> int:
        """verbosity"""
        return self.p.verbosity

    @verbosity.setter
    def verbosity(self, value: int):
        self.p.verbosity = value

    @property
    def control_vector_layer_start(self) -> int:
        """layer range for control vector"""
        return self.p.control_vector_layer_start

    @control_vector_layer_start.setter
    def control_vector_layer_start(self, value: int):
        self.p.control_vector_layer_start = value

    @property
    def control_vector_layer_end(self) -> int:
        """layer range for control vector"""
        return self.p.control_vector_layer_end

    @control_vector_layer_end.setter
    def control_vector_layer_end(self, value: int):
        self.p.control_vector_layer_end = value

    @property
    def ppl_stride(self) -> int:
        """stride for perplexity calculations. If left at 0, the pre-existing approach will be used."""
        return self.p.ppl_stride

    @ppl_stride.setter
    def ppl_stride(self, value: int):
        self.p.ppl_stride = value

    @property
    def ppl_output_type(self) -> int:
        """0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line

        (which is more convenient to use for plotting)
        """
        return self.p.ppl_output_type

    @ppl_output_type.setter
    def ppl_output_type(self, value: int):
        self.p.ppl_output_type = value

    @property
    def hellaswag(self) -> bool:
        """compute HellaSwag score over random tasks from datafile supplied in prompt"""
        return self.p.hellaswag

    @hellaswag.setter
    def hellaswag(self, value: bool):
        self.p.hellaswag = value

    @property
    def hellaswag_tasks(self) -> int:
        """number of tasks to use when computing the HellaSwag score"""
        return self.p.hellaswag_tasks

    @hellaswag_tasks.setter
    def hellaswag_tasks(self, value: int):
        self.p.hellaswag_tasks = value

    @property
    def winogrande(self) -> bool:
        """compute Winogrande score over random tasks from datafile supplied in prompt"""
        return self.p.winogrande

    @winogrande.setter
    def winogrande(self, value: bool):
        self.p.winogrande = value

    @property
    def winogrande_tasks(self) -> int:
        """number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed"""
        return self.p.winogrande_tasks

    @winogrande_tasks.setter
    def winogrande_tasks(self, value: int):
        self.p.winogrande_tasks = value

    @property
    def multiple_choice(self) -> bool:
        """compute TruthfulQA score over random tasks from datafile supplied in prompt"""
        return self.p.multiple_choice

    @multiple_choice.setter
    def multiple_choice(self, value: bool):
        self.p.multiple_choice = value

    @property
    def multiple_choice_tasks(self) -> int:
        """number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed"""
        return self.p.multiple_choice_tasks

    @multiple_choice_tasks.setter
    def multiple_choice_tasks(self, value: int):
        self.p.multiple_choice_tasks = value

    @property
    def kl_divergence(self) -> bool:
        """compute KL divergence"""
        return self.p.kl_divergence

    @kl_divergence.setter
    def kl_divergence(self, value: bool):
        self.p.kl_divergence = value

    @property
    def usage(self) -> bool:
        """print usage"""
        return self.p.usage

    @usage.setter
    def usage(self, value: bool):
        self.p.usage = value

    @property
    def use_color(self) -> bool:
        """use color to distinguish generations and inputs"""
        return self.p.use_color

    @use_color.setter
    def use_color(self, value: bool):
        self.p.use_color = value

    @property
    def special(self) -> bool:
        """enable special token output"""
        return self.p.special

    @special.setter
    def special(self, value: bool):
        self.p.special = value

    @property
    def interactive(self) -> bool:
        """interactive mode"""
        return self.p.interactive

    @interactive.setter
    def interactive(self, value: bool):
        self.p.interactive = value

    @property
    def interactive_first(self) -> bool:
        """wait for user input immediately"""
        return self.p.interactive_first

    @interactive_first.setter
    def interactive_first(self, value: bool):
        self.p.interactive_first = value

    # @property
    # def conversation(self) -> bool:
    #     """conversation mode (does not print special tokens and suffix/prefix)"""
    #     return self.p.conversation

    # @conversation.setter
    # def conversation(self, value: bool):
    #     self.p.conversation = value

    @property
    def prompt_cache_all(self) -> bool:
        """save user input and generations to prompt cache"""
        return self.p.prompt_cache_all

    @prompt_cache_all.setter
    def prompt_cache_all(self, value: bool):
        self.p.prompt_cache_all = value

    @property
    def prompt_cache_ro(self) -> bool:
        """open the prompt cache read-only and do not update it"""
        return self.p.prompt_cache_ro

    @prompt_cache_ro.setter
    def prompt_cache_ro(self, value: bool):
        self.p.prompt_cache_ro = value

    # @property
    # def use_color(self) -> bool:
    #     """use color to distinguish generations and inputs"""
    #     return self.p.use_color

    # @use_color.setter
    # def use_color(self, value: bool):
    #     self.p.use_color = value

    # @property
    # def special(self) -> bool:
    #     """enable special token output"""
    #     return self.p.special

    # @special.setter
    # def special(self, value: bool):
    #     self.p.special = value

    # @property
    # def interactive(self) -> bool:
    #     """interactive mode"""
    #     return self.p.interactive

    # @interactive.setter
    # def interactive(self, value: bool):
    #     self.p.interactive = value

    # @property
    # def interactive_first(self) -> bool:
    #     """wait for user input immediately"""
    #     return self.p.interactive_first

    # @interactive_first.setter
    # def interactive_first(self, value: bool):
    #     self.p.interactive_first = value

    # @property
    # def conversation(self) -> bool:
    #     """conversation mode (does not print special tokens and suffix/prefix)"""
    #     return self.p.conversation

    # @conversation.setter
    # def conversation(self, value: bool):
    #     self.p.conversation = value

    # @property
    # def prompt_cache_all(self) -> bool:
    #     """save user input and generations to prompt cache"""
    #     return self.p.prompt_cache_all

    # @prompt_cache_all.setter
    # def prompt_cache_all(self, value: bool):
    #     self.p.prompt_cache_all = value

    # @property
    # def prompt_cache_ro(self) -> bool:
    #     """ open the prompt cache read-only and do not update it"""
    #     return self.p.prompt_cache_ro

    # @prompt_cache_ro.setter
    # def prompt_cache_ro(self, value: bool):
    #     self.p.prompt_cache_ro = value

    @property
    def escape(self) -> bool:
        """escape special characters"""
        return self.p.escape

    @escape.setter
    def escape(self, value: bool):
        self.p.escape = value

    @property
    def multiline_input(self) -> bool:
        """reverse the usage of "\""""
        return self.p.multiline_input

    @multiline_input.setter
    def multiline_input(self, value: bool):
        self.p.multiline_input = value

    @property
    def simple_io(self) -> bool:
        """improves compatibility with subprocesses and limited consoles"""
        return self.p.simple_io

    @simple_io.setter
    def simple_io(self, value: bool):
        self.p.simple_io = value

    @property
    def cont_batching(self) -> bool:
        """insert new sequences for decoding on-the-fly"""
        return self.p.cont_batching

    @cont_batching.setter
    def cont_batching(self, value: bool):
        self.p.cont_batching = value

    @property
    def flash_attn(self) -> bool:
        """flash attention"""
        return self.p.flash_attn

    @flash_attn.setter
    def flash_attn(self, value: bool):
        self.p.flash_attn = value

    @property
    def no_perf(self) -> bool:
        """disable performance metrics"""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, value: bool):
        self.p.no_perf = value

    @property
    def ctx_shift(self) -> bool:
        """context shift on inifinite text generation"""
        return self.p.ctx_shift

    @ctx_shift.setter
    def ctx_shift(self, value: bool):
        self.p.ctx_shift = value

    @property
    def input_prefix_bos(self) -> bool:
        """prefix BOS to user inputs, preceding input_prefix"""
        return self.p.input_prefix_bos

    @input_prefix_bos.setter
    def input_prefix_bos(self, value: bool):
        self.p.input_prefix_bos = value

    @property
    def use_mmap(self) -> bool:
        """use mmap for faster loads"""
        return self.p.use_mmap

    @use_mmap.setter
    def use_mmap(self, value: bool):
        self.p.use_mmap = value

    @property
    def use_mlock(self) -> bool:
        """use mlock to keep model in memory"""
        return self.p.use_mlock

    @use_mlock.setter
    def use_mlock(self, value: bool):
        self.p.use_mlock = value

    @property
    def verbose_prompt(self) -> bool:
        """print prompt tokens before generation"""
        return self.p.verbose_prompt

    @verbose_prompt.setter
    def verbose_prompt(self, value: bool):
        self.p.verbose_prompt = value

    @property
    def display_prompt(self) -> bool:
        """print prompt before generation"""
        return self.p.display_prompt

    @display_prompt.setter
    def display_prompt(self, value: bool):
        self.p.display_prompt = value

    @property
    def dump_kv_cache(self) -> bool:
        """dump the KV cache contents for debugging purposes"""
        return self.p.dump_kv_cache

    @dump_kv_cache.setter
    def dump_kv_cache(self, value: bool):
        self.p.dump_kv_cache = value

    @property
    def no_kv_offload(self) -> bool:
        """disable KV offloading"""
        return self.p.no_kv_offload

    @no_kv_offload.setter
    def no_kv_offload(self, value: bool):
        self.p.no_kv_offload = value

    @property
    def warmup(self) -> bool:
        """warmup run"""
        return self.p.warmup

    @warmup.setter
    def warmup(self, value: bool):
        self.p.warmup = value

    @property
    def check_tensors(self) -> bool:
        """validate tensor data"""
        return self.p.check_tensors

    @check_tensors.setter
    def check_tensors(self, value: bool):
        self.p.check_tensors = value

    @property
    def cache_type_k(self) -> ggml_type:
        """data type for K cache"""
        return <ggml_type>self.p.cache_type_k

    @cache_type_k.setter
    def cache_type_k(self, ggml_type value):
        self.p.cache_type_k = value

    @property
    def cache_type_v(self) -> ggml_type:
        """data type for V cache"""
        return <ggml_type>self.p.cache_type_v

    @cache_type_v.setter
    def cache_type_v(self, ggml_type value):
        self.p.cache_type_v = value

    # @property
    # def mmproj(self) -> str:
    #     """path to multimodal projector"""
    #     return self.p.mmproj.decode()

    # @mmproj.setter
    # def mmproj(self, value: str):
    #     self.p.mmproj = value.encode('utf8')

    @property
    def image(self) -> list[str]:
        """paths to image file(s)"""
        result = []
        for i in range(self.p.image.size()):
            result.append(self.p.image[i].decode())
        return result

    @image.setter
    def image(self, files: list[str]):
        self.p.image.clear()
        for i in files:
            self.p.image.push_back(i.encode('utf8'))

    @property
    def embedding(self) -> bool:
        """get only sentence embedding"""
        return self.p.embedding

    @embedding.setter
    def embedding(self, value: bool):
        self.p.embedding = value

    @property
    def embd_normalize(self) -> int:
        """normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)"""
        return self.p.embd_normalize

    @embd_normalize.setter
    def embd_normalize(self, value: int):
        self.p.embd_normalize = value

    @property
    def embd_out(self) -> str:
        """empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix"""
        return self.p.embd_out.decode()

    @embd_out.setter
    def embd_out(self, value: str):
        self.p.embd_out = value.encode('utf8')

    @property
    def embd_sep(self) -> str:
        """separator of embendings"""
        return self.p.embd_sep.decode()

    @embd_sep.setter
    def embd_sep(self, value: str):
        self.p.embd_sep = value.encode('utf8')

    @property
    def reranking(self) -> bool:
        """enable reranking support on server"""
        return self.p.reranking

    @reranking.setter
    def reranking(self, value: bool):
        self.p.reranking = value

    @property
    def port(self) -> int:
        """server listens on this network port"""
        return self.p.port

    @port.setter
    def port(self, value: int):
        self.p.port = value

    @property
    def timeout_read(self) -> int:
        """http read timeout in seconds"""
        return self.p.timeout_read

    @timeout_read.setter
    def timeout_read(self, value: int):
        self.p.timeout_read = value

    @property
    def timeout_write(self) -> int:
        """http write timeout in seconds"""
        return self.p.timeout_write

    @timeout_write.setter
    def timeout_write(self, value: int):
        self.p.timeout_write = value

    @property
    def n_threads_http(self) -> int:
        """number of threads to process HTTP requests (TODO: support threadpool)"""
        return self.p.n_threads_http

    @n_threads_http.setter
    def n_threads_http(self, value: int):
        self.p.n_threads_http = value

    @property
    def n_cache_reuse(self) -> int:
        """min chunk size to reuse from the cache via KV shifting"""
        return self.p.n_cache_reuse

    @n_cache_reuse.setter
    def n_cache_reuse(self, value: int):
        self.p.n_cache_reuse = value

    @property
    def hostname(self) -> str:
        """server hostname"""
        return self.p.hostname.decode()

    @hostname.setter
    def hostname(self, value: str):
        self.p.hostname = value.encode('utf8')

    @property
    def public_path(self) -> str:
        """server public_path"""
        return self.p.public_path.decode()

    @public_path.setter
    def public_path(self, value: str):
        self.p.public_path = value.encode('utf8')

    @property
    def chat_template(self) -> str:
        """chat template"""
        return self.p.chat_template.decode()

    @chat_template.setter
    def chat_template(self, value: str):
        self.p.chat_template = value.encode('utf8')

    # @property
    # def system_prompt(self) -> str:
    #     """system prompt"""
    #     return self.p.system_prompt.decode()

    # @system_prompt.setter
    # def system_prompt(self, value: str):
    #     self.p.system_prompt = value.encode('utf8')

    @property
    def enable_chat_template(self) -> bool:
        """enable chat template"""
        return self.p.enable_chat_template

    @enable_chat_template.setter
    def enable_chat_template(self, value: bool):
        self.p.enable_chat_template = value

    @property
    def api_keys(self) -> list[str]:
        """list of api keys"""
        result = []
        for i in range(self.p.api_keys.size()):
            result.append(self.p.api_keys[i].decode())
        return result

    @api_keys.setter
    def api_keys(self, files: list[str]):
        self.p.api_keys.clear()
        for i in files:
            self.p.api_keys.push_back(i.encode('utf8'))

    @property
    def ssl_file_key(self) -> str:
        """ssl file key"""
        return self.p.ssl_file_key.decode()

    @ssl_file_key.setter
    def ssl_file_key(self, value: str):
        self.p.ssl_file_key = value.encode('utf8')

    @property
    def ssl_file_cert(self) -> str:
        """ssl file cert"""
        return self.p.ssl_file_cert.decode()

    @ssl_file_cert.setter
    def ssl_file_cert(self, value: str):
        self.p.ssl_file_cert = value.encode('utf8')

    @property
    def webui(self) -> bool:
        """enable webui"""
        return self.p.webui

    @webui.setter
    def webui(self, value: bool):
        self.p.webui = value

    @property
    def endpoint_slots(self) -> bool:
        """endpoint slots"""
        return self.p.endpoint_slots

    @endpoint_slots.setter
    def endpoint_slots(self, value: bool):
        self.p.endpoint_slots = value

    @property
    def endpoint_props(self) -> bool:
        """endpoint props"""
        return self.p.endpoint_props

    @endpoint_props.setter
    def endpoint_props(self, value: bool):
        self.p.endpoint_props = value

    @property
    def endpoint_metrics(self) -> bool:
        """endpoint metrics"""
        return self.p.endpoint_metrics

    @endpoint_metrics.setter
    def endpoint_metrics(self, value: bool):
        self.p.endpoint_metrics = value

    @property
    def log_json(self) -> bool:
        """log json"""
        return self.p.log_json

    @log_json.setter
    def log_json(self, value: bool):
        self.p.log_json = value

    @property
    def slot_save_path(self) -> str:
        """slot save path"""
        return self.p.slot_save_path.decode()

    @slot_save_path.setter
    def slot_save_path(self, value: str):
        self.p.slot_save_path = value.encode('utf8')

    @property
    def slot_prompt_similarity(self) -> float:
        """slot prompt similarity."""
        return self.p.slot_prompt_similarity

    @slot_prompt_similarity.setter
    def slot_prompt_similarity(self, value: float):
        self.p.slot_prompt_similarity = value

    @property
    def is_pp_shared(self) -> bool:
        """batched-bench params"""
        return self.p.is_pp_shared

    @is_pp_shared.setter
    def is_pp_shared(self, value: bool):
        self.p.is_pp_shared = value

    @property
    def n_pp(self) -> list[int]:
        return self.p.n_pp

    @n_pp.setter
    def n_pp(self, list[int] values):
        self.p.n_pp = values

    @property
    def n_tg(self) -> list[int]:
        return self.p.n_tg

    @n_tg.setter
    def n_tg(self, list[int] values):
        self.p.n_tg = values

    @property
    def n_pl(self) -> list[int]:
        return self.p.n_pl

    @n_pl.setter
    def n_pl(self, list[int] values):
        self.p.n_pl = values

    @property
    def context_files(self) -> list[str]:
        """context files to embed"""
        return [name.decode() for name in self.p.context_files]

    @context_files.setter
    def context_files(self, list[str] values):
        self.p.context_files = [name.encode() for name in values]

    @property
    def chunk_size(self) -> int:
        """chunk size for context embedding"""
        return self.p.chunk_size

    @chunk_size.setter
    def chunk_size(self, value: int):
        self.p.chunk_size = value

    @property
    def chunk_separator(self) -> str:
        """chunk separator for context embedding"""
        return self.p.chunk_separator.decode()

    @chunk_separator.setter
    def chunk_separator(self, value: str):
        self.p.chunk_separator = value.encode('utf8')

    @property
    def n_junk(self) -> int:
        """number of times to repeat the junk text"""
        return self.p.n_junk

    @n_junk.setter
    def n_junk(self, value: int):
        self.p.n_junk = value

    @property
    def i_pos(self) -> int:
        """position of the passkey in the junk text"""
        return self.p.i_pos

    @i_pos.setter
    def i_pos(self, value: int):
        self.p.i_pos = value

    @property
    def out_file(self) -> str:
        """save the resulting imatrix to this file"""
        return self.p.out_file.decode()

    @out_file.setter
    def out_file(self, value: str):
        self.p.out_file = value.encode('utf8')

    @property
    def n_out_freq(self) -> int:
        """output the imatrix every n_out_freq iterations"""
        return self.p.n_out_freq

    @n_out_freq.setter
    def n_out_freq(self, value: int):
        self.p.n_out_freq = value

    @property
    def n_save_freq(self) -> int:
        """save the imatrix every n_save_freq iterations"""
        return self.p.n_save_freq

    @n_save_freq.setter
    def n_save_freq(self, value: int):
        self.p.n_save_freq = value

    @property
    def i_chunk(self) -> int:
        """start processing from this chunk"""
        return self.p.i_chunk

    @i_chunk.setter
    def i_chunk(self, value: int):
        self.p.i_chunk = value

    @property
    def process_output(self) -> bool:
        """collect data for the output tensor"""
        return self.p.process_output

    @process_output.setter
    def process_output(self, value: bool):
        self.p.process_output = value

    @property
    def compute_ppl(self) -> bool:
        """whether to compute perplexity"""
        return self.p.compute_ppl

    @compute_ppl.setter
    def compute_ppl(self, value: bool):
        self.p.compute_ppl = value

    @property
    def n_pca_batch(self) -> int:
        """start processing from this chunk"""
        return self.p.n_pca_batch

    @n_pca_batch.setter
    def n_pca_batch(self, value: int):
        self.p.n_pca_batch = value

    @property
    def n_pca_iterations(self) -> int:
        """start processing from this chunk"""
        return self.p.n_pca_iterations

    @n_pca_iterations.setter
    def n_pca_iterations(self, value: int):
        self.p.n_pca_iterations = value

    # // cvector-generator params
    # dimre_method cvector_dimre_method = DIMRE_METHOD_PCA;
    # std::string cvector_outfile       =
    # std::string cvector_positive_file = "examples/cvector-generator/positive.txt";
    # std::string cvector_negative_file = "examples/cvector-generator/negative.txt";

    # bool spm_infill = false; // suffix/prefix/middle pattern for infill

    # std::string lora_outfile = "ggml-lora-merged-f16.gguf";

    # // batched-bench params
    # bool batched_bench_output_jsonl = false;


cdef class LlamaModelParams:
    cdef llama_cpp.llama_model_params p

    # NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
    # cdef ggml_backend_dev_t * devices

    def __init__(self):
        self.p = llama_cpp.llama_model_default_params()
        # self.p.progress_callback = &progress_callback # FIXME: causes crash

    @staticmethod
    cdef LlamaModelParams from_instance(llama_cpp.llama_model_params params):
        cdef LlamaModelParams wrapper = LlamaModelParams.__new__(LlamaModelParams)
        wrapper.p = params
        return wrapper

    # @property
    # def tensor_buft_overrides(self) -> LlamaModelTensorBuftOverride:
    #     """NULL-terminated list of buffer types to use for tensors that match a pattern"""
    #     return self.p.tensor_buft_overrides

    @property
    def n_gpu_layers(self) -> int:
        """Number of layers to store in VRAM."""
        return self.p.n_gpu_layers

    @n_gpu_layers.setter
    def n_gpu_layers(self, value: int):
        self.p.n_gpu_layers = value

    @property
    def split_mode(self) -> llama_split_mode:
        """How to split the model across multiple GPUs."""
        return llama_split_mode(self.p.split_mode)

    @split_mode.setter
    def split_mode(self, llama_split_mode value):
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
        cdef size_t length = sizeof(self.p.tensor_split)
        results = []
        for i in range(length):
            results.append(self.p.tensor_split[i])
        return results

    # @tensor_split.setter
    # def tensor_split(self, value: list[float]):
    #     cdef size_t size = llama_cpp.llama_max_devices()
    #     for i in range(size):
    #         self.p.tensor_split[i] = value[i]


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

    # @property
    # def kv_overrides(self) -> list[str]:
    #     """override key-value pairs of the model meta data

    #     const llama_model_kv_override * kv_overrides
    #     """
    #     return self.p.kv_overrides

    # @kv_overrides.setter
    # def kv_overrides(self, value: list[str]):
    #     self.p.kv_overrides = value

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


cdef class LlamaModelQuantizeParams:
    cdef llama_cpp.llama_model_quantize_params p

    def __init__(self):
        self.p = llama_cpp.llama_model_quantize_default_params()

    @staticmethod
    cdef LlamaModelQuantizeParams from_instance(llama_cpp.llama_model_quantize_params params):
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
    def ftype(self) -> int:
        """quantize to this llama_ftype"""
        return llama_ftype(self.p.ftype)

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



cdef class LlamaVocab:
    """cython wrapper for llama_cpp.llama_vocab"""
    cdef llama_cpp.llama_vocab * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    @staticmethod
    cdef LlamaVocab from_ptr(llama_cpp.llama_vocab *ptr, bint owner=False):
        cdef LlamaVocab wrapper = LlamaVocab.__new__(LlamaVocab)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @property
    def vocab_type(self) -> llama_vocab_type:
        return llama_vocab_type(llama_cpp.llama_get_vocab_type(self.ptr))

    @property
    def n_vocab(self) -> int:
        return llama_cpp.llama_vocab_n_tokens(self.ptr)

    def get_text(self, llama_cpp.llama_token token) -> str:
        return llama_cpp.llama_vocab_get_text(self.ptr, token).decode("utf-8")

    def get_score(self, llama_cpp.llama_token token) -> float:
        return llama_cpp.llama_vocab_get_score(self.ptr, token)

    def get_attr(self, llama_cpp.llama_token token) -> llama_token_attr:
        return llama_cpp.llama_vocab_get_attr(self.ptr, token)

    def is_eog(self, llama_cpp.llama_token token) -> bool:
        """Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)"""
        return llama_cpp.llama_vocab_is_eog(self.ptr, token)

    def is_control(self, llama_cpp.llama_token token) -> bool:
        """Identify if Token Id is a control token or a render-able token"""
        return llama_cpp.llama_vocab_is_control(self.ptr, token)

    # Special tokens

    def token_bos(self) -> int:
        """beginning-of-sentence"""
        return llama_cpp.llama_vocab_bos(self.ptr)

    def token_eos(self) -> int:
        """end-of-sentence"""
        return llama_cpp.llama_vocab_eos(self.ptr)

    def token_eot(self) -> int:
        """end-of-turn"""
        return llama_cpp.llama_vocab_eot(self.ptr)

    def token_sep(self) -> int:
        """sentence separator"""
        return llama_cpp.llama_vocab_sep(self.ptr)

    def token_nl(self) -> int:
        """next-line"""
        return llama_cpp.llama_vocab_nl(self.ptr)

    def token_pad(self) -> int:
        """padding"""
        return llama_cpp.llama_vocab_pad(self.ptr)

    def get_add_bos(self) -> bool:
        """add beginning-of-sentence token"""
        return llama_cpp.llama_vocab_get_add_bos(self.ptr)

    def get_add_eos(self) -> bool:
        """add end-of-sentence token"""
        return llama_cpp.llama_vocab_get_add_eos(self.ptr)

    # infill tokens

    def fim_prefix(self) -> int:
        return llama_cpp.llama_vocab_fim_pre(self.ptr)

    def fim_middle(self) -> int:
        return llama_cpp.llama_vocab_fim_suf(self.ptr)

    def fim_suffix(self) -> int:
        return llama_cpp.llama_vocab_fim_mid(self.ptr)

    def fim_pad(self) -> int:
        return llama_cpp.llama_vocab_fim_pad(self.ptr)

    def fim_rep(self) -> int:
        return llama_cpp.llama_vocab_fim_rep(self.ptr)

    def fim_sep(self) -> int:
        return llama_cpp.llama_vocab_fim_sep(self.ptr)

    # Tokenization

    # def tokenize(self, text: str, add_special: bool, parse_special: bool) -> list[int]:
    #     """Convert the provided text into tokens.

    #     text: string to be converted into token.
    #     add_special: Allow to add BOS and EOS tokens if model is configured to do so.
    #     parse_special: Allow tokenizing special and/or control tokens which otherwise
    #                    are not exposed and treated as plaintext. Does not insert a leading space.
    #     Returns the number of tokens on success, no more than n_tokens_max
    #     Returns a negative number on failure - the number of tokens that would have been returned
    #     """
    #     cdef int n_ctx = self.n_ctx_train()
    #     cdef vector[llama_cpp.llama_token] tokens
    #     tokens.reserve(n_ctx)
    #     n_tokens = llama_cpp.llama_tokenize(
    #         self.ptr, text.encode(), len(text), tokens.data(), n_ctx, add_special, parse_special
    #     )
    #     if n_tokens < 0:
    #         raise RuntimeError(
    #             f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
    #         )
    #     return tokens[:n_tokens]

    def tokenize(self, text: str, add_special: bool, parse_special: bool) -> list[int]:
        """Convert the provided text into tokens.

        text: string to be converted into token.
        add_special: Allow to add BOS and EOS tokens if model is configured to do so.
        parse_special: Allow tokenizing special and/or control tokens which otherwise
                       are not exposed and treated as plaintext. Does not insert a leading space.
        Returns the number of tokens on success, no more than n_tokens_max
        Returns a negative number on failure - the number of tokens that would have been returned
        """
        cdef vector[llama_cpp.llama_token] tokens
        tokens.reserve(self.n_vocab)
        n_tokens = llama_cpp.llama_tokenize(
            self.ptr, text.encode(), len(text), tokens.data(), self.n_vocab, add_special, parse_special
        )
        if n_tokens < 0:
            raise RuntimeError(
                f'Failed to tokenize: text="{text}" n_tokens={n_tokens}'
            )
        return tokens[:n_tokens]

    def token_to_piece(self, token: int, lstrip: int = 0, special: bool = False) -> str:
        """Token Id -> Piece.

        special: If true, special tokens are rendered in the output.
        Uses the vocabulary in the provided context.
        Does not write null terminator to the buffer.
        User can skip up to 'lstrip' leading spaces before copying
        (useful when encoding/decoding multiple tokens with 'add_space_prefix')
        """
        cdef char buf[32]
        llama_cpp.llama_token_to_piece(self.ptr, token, buf, 32, lstrip, special)
        return buf.decode()

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
        cdef vector[int] vec

        for i in tokens:
            vec.push_back(i)

        cdef int32_t res = llama_cpp.llama_detokenize(
            self.ptr,
            <const llama_cpp.llama_token *>vec.data(),
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
    """cython wrapper for llama_cpp.llama_model."""
    cdef llama_cpp.llama_model * ptr
    cdef public LlamaModelParams params
    cdef public str path_model
    cdef public bint verbose
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = True

    def __init__(self, path_model: str, params: Optional[LlamaModelParams] = None, verbose: bool = True):
        self.path_model = path_model
        self.params = params if params else LlamaModelParams()
        self.verbose = verbose

        if not os.path.exists(path_model):
            raise ValueError(f"Model path does not exist: {path_model}")

        # with suppress_stdout_stderr(disable=verbose):
        self.ptr = llama_cpp.llama_model_load_from_file(
            self.path_model.encode("utf-8"),
            self.params.p
        )

        if self.ptr is NULL:
            raise ValueError(f"Failed to load model from file: {path_model}")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_model_free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaModel from_ptr(llama_cpp.llama_model *ptr, bint owner=False):
        cdef LlamaModel wrapper = LlamaModel.__new__(LlamaModel)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    @property
    def rope_type(self) -> llama_rope_type:
        return llama_rope_type(llama_cpp.llama_get_model_rope_type(self.ptr))

    @property
    def n_ctx_train(self) -> int:
        return llama_cpp.llama_model_n_ctx_train(self.ptr)

    @property
    def n_embd(self) -> int:
        return llama_cpp.llama_model_n_embd(self.ptr)

    @property
    def n_layer(self) -> int:
        return llama_cpp.llama_model_n_layer(self.ptr)

    @property
    def n_head(self) -> int:
        return llama_cpp.llama_model_n_head(self.ptr)

    @property
    def n_head_kv(self) -> int:
        return llama_cpp.llama_model_n_head_kv(self.ptr)

    @property
    def rope_freq_scale_train(self) -> float:
        """Get the model's RoPE frequency scaling factor"""
        return llama_cpp.llama_model_rope_freq_scale_train(self.ptr)

    @property
    def desc(self) -> str:
        """Get a string describing the model type"""
        cdef char buf[1024]
        llama_cpp.llama_model_desc(self.ptr, buf, 1024)
        return buf.decode("utf-8")

    @property
    def size(self) -> int:
        """Returns the total size of all the tensors in the model in bytes"""
        return <uint64_t>llama_cpp.llama_model_size(self.ptr)

    @property
    def n_params(self) -> int:
        """Returns the total number of parameters in the model"""
        return <uint64_t>llama_cpp.llama_model_n_params(self.ptr)

    # def get_tensor(self, name: str) -> GGMLTensor:
    #     """Get a llama model tensor"""
    #     cdef llama_cpp.ggml_tensor * tensor = llama_cpp.llama_get_model_tensor(
    #         self.ptr, name.encode("utf-8"))
    #     return GGMLTensor.from_ptr(tensor)

    # vocab

    def get_vocab(self) -> LlamaVocab:
        cdef llama_cpp.llama_vocab * vocab = <llama_cpp.llama_vocab *>llama_cpp.llama_model_get_vocab(self.ptr)
        return LlamaVocab.from_ptr(vocab)

    # sampling

    def sampler_init(self, CommonParamsSampling params) -> CommonSampler:
        """initialize common_sampler"""
        return CommonSampler(self, params)

    # lora

    def lora_adapter_init(self, str path_lora) -> LlamaAdapterLora:
        """Load a LoRA adapter from file

        The loaded adapter will be associated to the given model, and will be free when the model is deleted
        """
        cdef llama_cpp.llama_adapter_lora * ptr = llama_cpp.llama_adapter_lora_init(
            self.ptr, path_lora.encode())
        cdef LlamaAdapterLora adapter = LlamaAdapterLora.from_ptr(ptr)
        return adapter

    # metadata

    def meta_val_str(self, str key) -> str:
        """Get metadata value as a string by key name"""
        cdef char buf[128]
        cdef int res = llama_cpp.llama_model_meta_val_str(self.ptr, key.encode(), buf, 128)
        if res == -1:
            raise ValueError(F"could not get metadata value from {key}")
        cdef str value = buf.decode('UTF-8')
        return value

    def meta_count(self):
        """Get the number of metadata key/value pairs"""
        return llama_cpp.llama_model_meta_count(self.ptr)

    def meta_key_by_index(self, int index) -> str:
        """Get metadata key name by index"""
        cdef char buf[128]
        cdef int res = llama_cpp.llama_model_meta_key_by_index(self.ptr, index, buf, 128)
        cdef str key = buf.decode('UTF-8')
        return key

    def meta_val_str_by_index(self, int index) -> str:
        """Get metadata key name by index"""
        cdef char buf[128]
        cdef int res = llama_cpp.llama_model_meta_val_str_by_index(self.ptr, index, buf, 128)
        cdef str value = buf.decode('UTF-8')
        return value

    # encode / decode

    def has_encoder(self) -> bool:
        """Returns true if the model contains an encoder that requires llama_encode() call"""
        return llama_cpp.llama_model_has_encoder(self.ptr)

    def has_decoder(self) -> bool:
        """Returns true if the model contains a decoder that requires llama_decode() callD"""
        return llama_cpp.llama_model_has_decoder(self.ptr)

    def decoder_start_token(self) -> int:
        """For encoder-decoder models, this function returns id of the token that must be provided
        to the decoder to start generating output sequence. For other models, it returns -1.
        """
        return llama_cpp.llama_model_decoder_start_token(self.ptr)

    def is_recurrent(self) -> bool:
        """Returns true if the model is recurrent (like Mamba, RWKV, etc.)"""
        return llama_cpp.llama_model_is_recurrent(self.ptr)

    # chat template

    def chat_apply_template(self, str tmpl, list[LlamaChatMessage] msgs, bint add_assistant_msg) -> str:
        """Apply chat template. Inspired by hf apply_chat_template() on python.

        Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
        NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
        @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
        @param chat Pointer to a list of multiple llama_chat_message
        @param n_msg Number of llama_chat_message in this chat
        @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
        @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
        @param length The size of the allocated buffer
        @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
        """
        cdef vector[llama_cpp.llama_chat_message] vec
        cdef char * buf = NULL
        cdef int length = 0
        for i in range(len(msgs)):
            vec.push_back(msgs[i].p)
        cdef int n_bytes = llama_cpp.llama_chat_apply_template(
            tmpl.encode(),
            vec.data(),
            vec.size(),
            add_assistant_msg,
            buf,
            length
        )
        cdef str result = buf.decode()
        free(buf)
        return result

    # def get_builtin_chat_template(self) -> str:
    #     """Get the built-in chat template for the model.

    #     Return empty string if not present."""
    #     return llama_cpp.common_get_builtin_chat_template(self.ptr).decode()


    # Extra

    def metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        buffer_size = 1024
        cdef int nbytes
        cdef char * buffer = <char*>calloc(buffer_size, sizeof(char))
        assert self.ptr is not NULL
        # iterate over model keys
        for i in range(llama_cpp.llama_model_meta_count(self.ptr)):
            nbytes = llama_cpp.llama_model_meta_key_by_index(
                self.ptr, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama_cpp.llama_model_meta_key_by_index(
                    self.ptr, i, buffer, buffer_size
                )
            key = buffer.decode("utf-8")
            nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                self.ptr, i, buffer, buffer_size
            )
            if nbytes > buffer_size:
                buffer_size = nbytes + 1
                buffer = <char*>realloc(buffer, buffer_size * sizeof(char));
                nbytes = llama_cpp.llama_model_meta_val_str_by_index(
                    self.ptr, i, buffer, buffer_size
                )
            value = buffer.decode("utf-8")
            metadata[key] = value
        free(buffer)
        return metadata

    @staticmethod
    def default_params() -> LlamaModelParams:
        """Get the default llama_model_params."""
        # return llama_cpp.llama_model_default_params()
        return LlamaModelParams()



def chat_builtin_templates() -> list[str]:
    """Get list of built-in chat templates"""
    cdef vector[const char *] supported_tmpl
    cdef int32_t res = llama_cpp.llama_chat_builtin_templates(NULL, 0)
    assert res > 0
    supported_tmpl.resize(res)
    res = llama_cpp.llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size())
    return [name.decode() for name in supported_tmpl]


cdef class LlamaContextParams:
    cdef llama_cpp.llama_context_params p

    def __init__(self):
        self.p = llama_cpp.llama_context_default_params()

    @staticmethod
    cdef LlamaContextParams from_common_params(CommonParams params):
        cdef LlamaContextParams wrapper = LlamaContextParams.__new__(LlamaContextParams)
        wrapper.p = llama_cpp.common_context_params_to_llama(params.p)
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
    def rope_scaling_type(self) -> llama_rope_scaling_type:
        """number of threads to use for batch processing"""
        return <llama_rope_scaling_type>self.p.rope_scaling_type

    @rope_scaling_type.setter
    def rope_scaling_type(self, llama_rope_scaling_type value):
        self.p.rope_scaling_type = value

    @property
    def pooling_type(self) -> llama_pooling_type:
        """whether to pool (sum) embedding results by sequence id"""
        return <llama_pooling_type>self.p.pooling_type

    @pooling_type.setter
    def pooling_type(self, llama_pooling_type value):
        self.p.pooling_type = value

    @property
    def attention_type(self) -> llama_attention_type:
        """attention type to use for embeddings"""
        return <llama_attention_type>self.p.attention_type

    @attention_type.setter
    def attention_type(self, llama_attention_type value):
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

    @property
    def defrag_thold(self) -> float:
        """KV cache defragmentation threshold."""
        return self.p.defrag_thold

    @defrag_thold.setter
    def defrag_thold(self, value: float):
        self.p.defrag_thold = value

    # ggml_backend_sched_eval_callback cb_eval;

    # @property
    # def cb_eval(self) -> py_sched_eval_callback:
    #     """get/set python ggml backend sched eval callback."""
    #     return <object>self.p.cb_eval_user_data

    # @cb_eval.setter
    # def cb_eval(self, object py_sched_eval_callback):
    #     self.p.cb_eval_user_data = <void*>py_sched_eval_callback

    @property
    def type_k(self) -> llama_cpp.ggml_type:
        """data type for K cache"""
        return <llama_cpp.ggml_type>self.p.type_k

    @type_k.setter
    def type_k(self, llama_cpp.ggml_type value):
        self.p.type_k = value

    @property
    def type_v(self) -> llama_cpp.ggml_type:
        """data type for V cache"""
        return <llama_cpp.ggml_type>self.p.type_v

    @type_v.setter
    def type_v(self, llama_cpp.ggml_type value):
        self.p.type_v = value

    # @property
    # def logits_all(self) -> bool:
    #     """the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)"""
    #     return self.p.logits_all

    # @logits_all.setter
    # def logits_all(self, value: bool):
    #     self.p.logits_all = value

    @property
    def embeddings(self) -> bool:
        """if true, extract embeddings (together with logits)"""
        return self.p.embeddings

    @embeddings.setter
    def embeddings(self, value: bool):
        self.p.embeddings = value

    @property
    def offload_kqv(self) -> bool:
        """whether to offload the KQV ops (including the KV cache) to GPU"""
        return self.p.offload_kqv

    @offload_kqv.setter
    def offload_kqv(self, value: bool):
        self.p.offload_kqv = value

    @property
    def flash_attn(self) -> bool:
        """whether to use flash attention [EXPERIMENTAL]"""
        return self.p.flash_attn

    @flash_attn.setter
    def flash_attn(self, value: bool):
        self.p.flash_attn = value

    @property
    def no_perf(self) -> bool:
        """whether to measure performance timings"""
        return self.p.no_perf

    @no_perf.setter
    def no_perf(self, value: bool):
        self.p.no_perf = value

    # ggml_abort_callback abort_callback;
    # void *              abort_callback_data;


cdef class LlamaContext:
    """Intermediate Python wrapper for a llama.cpp llama_context."""
    cdef llama_cpp.llama_context * ptr
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

        self.ptr = llama_cpp.llama_init_from_model(self.model.ptr, self.params.p)

        if self.ptr is NULL:
            raise ValueError("Failed to create llama_context")

    def __dealloc__(self):
        if self.ptr is not NULL and self.owner is True:
            llama_cpp.llama_free(self.ptr)
            self.ptr = NULL

    @staticmethod
    cdef LlamaContext from_ptr(llama_cpp.llama_context *ptr, bint owner=False):
        cdef LlamaContext wrapper = LlamaContext.__new__(LlamaContext)
        wrapper.ptr = ptr
        wrapper.owner = owner
        return wrapper

    def close(self):
        self.__dealloc__()

    @property
    def n_ctx(self) -> int:
        return llama_cpp.llama_n_ctx(self.ptr)

    @property
    def n_batch(self) -> int:
        return llama_cpp.llama_n_batch(self.ptr)

    @property
    def n_ubatch(self) -> int:
        return llama_cpp.llama_n_ubatch(self.ptr)

    @property
    def n_seq_max(self) -> int:
        return llama_cpp.llama_n_seq_max(self.ptr)

    @property
    def pooling_type(self) -> int:
        return <llama_pooling_type>llama_cpp.llama_get_pooling_type(self.ptr)

    # Manage Threadpools
    # -------------------------------------------------------------------------

    def attach_threadpool(self, GGMLThreadPool threadpool, GGMLThreadPool threadpool_batch):
        llama_cpp.llama_attach_threadpool(self.ptr, threadpool.ptr, threadpool_batch.ptr)

    def detach_threadpool(self):
        llama_cpp.llama_detach_threadpool(self.ptr)

    # Lora
    # -------------------------------------------------------------------------

    def set_adapter_lora(self, LlamaAdapterLora adapter, float scale):
        """Add a loaded LoRA adapter to given context

        This will not modify model's weight
        """
        cdef int32_t res = llama_cpp.llama_set_adapter_lora(
            self.ptr, adapter.ptr, scale)
        if res == -1:
            raise ValueError(f"cannot load lora adapter to context")

    def rm_adapter_lora(self, LlamaAdapterLora adapter):
        """Remove a specific LoRA adapter from given context

        Return -1 if the adapter is not present in the context
        """
        cdef int32_t res = llama_cpp.llama_rm_adapter_lora(
            self.ptr, adapter.ptr)
        if res == -1:
            raise ValueError(f"cannot remove, lora the adapter is not present in the context")

    def clear_adapter_lora(self):
        """Remove all LoRA adapters from given context"""
        llama_cpp.llama_clear_adapter_lora(self.ptr)


    def apply_adapter_cvec(self, data: list[float], n_embd: int, il_start: int, il_end: int) -> int:
        """Apply a loaded control vector to a llama_context, or if data is NULL, clear
        the currently loaded vector.

        `n_embd` should be the size of a single layer's control, and data should point
        to an n_embd x n_layers buffer starting from layer 1.

        `il_start` and `il_end` are the layer range the vector should apply to (both inclusive)

        See llama_control_vector_load in common to load a control vector.
        """
        cdef vector[float] vec
        for i in data:
            vec.push_back(i)

        return llama_cpp.llama_apply_adapter_cvec(
            self.ptr,
            vec.data(),
            vec.size(),
            n_embd,
            il_start,
            il_end
        )

    # KV cache
    # -------------------------------------------------------------------------

    def kv_cache_clear(self):
        """Clear the KV cache - both cell info is erased and KV data is zeroed"""
        llama_cpp.llama_kv_self_clear(self.ptr)

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int):
        """Removes all tokens that belong to the specified sequence and have positions in [p0, p1)

        Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
        seq_id < 0 : match any sequence
        p0 < 0     : [0,  p1]
        p1 < 0     : [p0, inf)
        """
        llama_cpp.llama_kv_self_seq_rm(self.ptr, seq_id, p0, p1)

    def kv_cache_seq_cp(self, seq_id_src: int, seq_id_dst: int, p0: int, p1: int):
        """Copy all tokens that belong to the specified sequence to another sequence

        Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
        p0 < 0 : [0,  p1]
        p1 < 0 : [p0, inf)
        """
        llama_cpp.llama_kv_self_seq_cp(self.ptr, seq_id_src, seq_id_dst, p0, p1)

    def kv_cache_seq_keep(self, seq_id: int):
        """Removes all tokens that do not belong to the specified sequence"""
        llama_cpp.llama_kv_self_seq_keep(self.ptr, seq_id)

    def kv_cache_seq_add(self, seq_id: int, p0: int, p1: int, shift: int):
        """Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)

        If the KV cache is RoPEd, the KV data is updated accordingly:
          - lazily on next llama_decode()
          - explicitly with llama_kv_cache_update()
        p0 < 0 : [0,  p1]
        p1 < 0 : [p0, inf)
        """
        llama_cpp.llama_kv_self_seq_add(self.ptr, seq_id, p0, p1, shift)

    def kv_cache_seq_div(self, seq_id: int, p0: int, p1: int, d: int):
        """Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)

        If the KV cache is RoPEd, the KV data is updated accordingly:
          - lazily on next llama_decode()
          - explicitly with llama_kv_cache_update()
        p0 < 0 : [0,  p1]
        p1 < 0 : [p0, inf)
        """
        llama_cpp.llama_kv_self_seq_div(self.ptr, seq_id, p0, p1, d)

    def kv_cache_seq_pos_max(self, seq_id: int) -> int:
        """Returns the largest position present in the KV cache for the specified sequence"""
        return llama_cpp.llama_kv_self_seq_pos_max(self.ptr, seq_id)


    def kv_cache_defrag(self):
        """Defragment the KV cache

        This will be applied:
        - lazily on next llama_decode()
        - explicitly with llama_kv_cache_update()
        """
        llama_cpp.llama_kv_self_defrag(self.ptr)


    def kv_cache_update(self):
        """Apply the KV cache updates (such as K-shifts, defragmentation, etc.)"""
        llama_cpp.llama_kv_self_update(self.ptr)


    def kv_cache_can_shift(self) -> bool:
        """Check if the context supports KV cache shifting."""
        return <bint>llama_cpp.llama_kv_self_can_shift(self.ptr)

    # State / sessions
    # -------------------------------------------------------------------------

    def get_state_size(self) -> int:
        """Returns the *actual* size in bytes of the state

        (logits, embedding and kv_cache)
        Only use when saving the state, not when restoring it, otherwise the size may be too small.
        """
        return llama_cpp.llama_state_get_size(self.ptr)

    def get_state_data(self) -> list[int]:
        """Copies the state to the specified destination address.

        Destination needs to have allocated enough memory.
        Returns the number of bytes copied
        """
        cdef uint8_t * dst = NULL
        cdef size_t size = 0
        cdef vector[uint8_t] result
        cdef size_t copied = llama_cpp.llama_state_get_data(self.ptr, dst, size)
        for i in range(size):
            result.push_back(dst[i])
        return result

    def set_state_data(self, data: list[int]) -> int:
        """Set the state reading from the specified address

        Returns the number of bytes read
        """
        cdef vector[uint8_t] result = data
        cdef size_t read = llama_cpp.llama_state_set_data(self.ptr, result.data(), result.size())
        return read

    def load_state_file(self, path_session: str, max_n_tokens: int = 256) -> list[int]:
        """Load session file"""
        cdef llama_cpp.llama_token * tokens_out = NULL
        cdef size_t * n_token_count_out = NULL
        cdef bint loaded = llama_cpp.llama_state_load_file(
            self.ptr,
            path_session.encode(),
            tokens_out,
            max_n_tokens,
            n_token_count_out)
        cdef vector[int] result
        if loaded:
            for i in range(n_token_count_out[0]):
                result.push_back(tokens_out[i])
        return result

    def save_state_file(self, path_session: str, tokens: list[int]) -> bool:
        """Save session file"""
        cdef vector[llama_cpp.llama_token] vec_tokens
        for token in tokens:
            vec_tokens.push_back(<llama_cpp.llama_token>token)
        return llama_cpp.llama_state_save_file(
            self.ptr,
            path_session.encode(),
            vec_tokens.data(),
            vec_tokens.size())

    def get_state_seq_size(self, int seq_id) -> int:
        """Get the exact size needed to copy the KV cache of a single sequence"""
        return llama_cpp.llama_state_seq_get_size(self.ptr, seq_id)

    def get_state_seq_data(self, int seq_id) -> list[int]:
        """Copy the KV cache of a single sequence into the specified buffer"""
        cdef uint8_t * dst = NULL
        cdef size_t size = 0
        cdef size_t copied = llama_cpp.llama_state_seq_get_data(
            self.ptr, dst, size, seq_id)
        cdef vector[uint8_t] result
        if copied > 0:
            for i in range(size):
                result.push_back(dst[i])
        return result

    def set_state_seq_data(self, src: list[int], dest_seq_id: int):
        """Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence

        Returns:
         - Positive: Ok
         - Zero: Failed to load
        """
        cdef vector[int] vec
        cdef size_t res = 0
        for i in src:
            vec.push_back(i)
        res = llama_cpp.llama_state_seq_set_data(
            self.ptr, src.data(), src.size(), dest_seq_id)
        if res == 0:
            raise ValueError("Failed to load sequence data")

    def save_state_seq_file(self, filepath: str, seq_id: int, tokens: list[int]):
        """Save state sequence data to a file"""
        cdef vector[int] vec
        cdef size_t res = 0
        for i in tokens:
            vec.push_back(i)
        res = llama_cpp.llama_state_seq_save_file(
            self.ptr,
            filepath.encode(),
            seq_id,
            <const llama_cpp.llama_token *>vec.data(),
            vec.size())
        if res == 0:
            raise ValueError(f"Failed to save seq data {filepath}")

    def load_state_seq_file(self, filepath: str, dest_seq_id: int, max_n_tokens: int = 256):
        """Load state sequence data from a file"""
        cdef llama_cpp.llama_token * tokens_out = NULL
        cdef size_t * n_token_count_out = NULL
        cdef size_t loaded = llama_cpp.llama_state_seq_load_file(
            self.ptr,
            filepath.encode(),
            dest_seq_id,
            tokens_out,
            max_n_tokens,
            n_token_count_out)
        cdef vector[int] result
        if loaded:
            for i in range(n_token_count_out[0]):
                result.push_back(tokens_out[i])
        return result

    # Decoding
    # -------------------------------------------------------------------------

    def encode(self, LlamaBatch batch):
        """Processes a batch of tokens with the encoder part of the encoder-decoder model.

        Stores the encoder output internally for later use by the decoder cross-attention layers.
          0 - success
        < 0 - error
        """
        cdef int32_t res = llama_cpp.llama_encode(self.ptr, batch.p)
        if res < 0:
            raise RuntimeError("error encoding batch")

    def decode(self, LlamaBatch batch) -> int:
        """Positive return values does not mean a fatal error, but rather a warning.

          0 - success
          1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
        < 0 - error
        """
        cdef int32_t res = llama_cpp.llama_decode(self.ptr,batch.p)
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
        llama_cpp.llama_set_n_threads(self.ptr, n_threads, n_threads_batch)

    def n_threads(self):
        """Get the number of threads used for generation of a single token."""
        return llama_cpp.llama_n_threads(self.ptr)

    def n_threads_batch(self):
        """Get the number of threads used for prompt and batch processing (multiple token)."""
        return llama_cpp.llama_n_threads_batch(self.ptr)

    def set_embeddings_mode(self, embeddings: bool):
        """Set whether the model is in embeddings mode or not

        If true, embeddings will be returned but logits will not
        """
        llama_cpp.llama_set_embeddings(self.ptr, embeddings)

    def set_causal_attn(self, causal_attn: bool):
        """Set whether to use causal attention or not

        If set to true, the model will only attend to the past tokens
        """
        llama_cpp.llama_set_causal_attn(self.ptr, causal_attn)

    def set_abort_callback(self, object py_abort_callback):
        """Set abort callback"""
        llama_cpp.llama_set_abort_callback(self.ptr,
            <llama_cpp.ggml_abort_callback>&abort_callback, <void*>py_abort_callback)

    def synchronize(self):
        """Wait until all computations are finished

        This is automatically done when using one of the functions below to obtain the computation results
        and is not necessary to call it explicitly in most cases
        """
        llama_cpp.llama_synchronize(self.ptr)

    # def n_outputs(self) -> int:
    #     return llama_cpp.llama_n_outputs(self.ptr)

    def get_logits(self) -> list[float]:
        """Token logits obtained from the last call to llama_decode()

        The logits for which llama_batch.logits[i] != 0 are stored contiguously
        in the order they have appeared in the batch.

        Rows: number of tokens for which llama_batch.logits[i] != 0
        Cols: n_vocab
        """
        cdef int n_vocab = self.model.n_vocab
        cdef float * logits = llama_cpp.llama_get_logits(self.ptr)
        if logits is NULL:
            # TODO: should one just return [] here?
            raise ValueError('no logits available')
        cdef vector[float] vec
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
        cdef float * logits = llama_cpp.llama_get_logits_ith(self.ptr, i)
        cdef vector[float] vec
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
        cdef float * embds = llama_cpp.llama_get_embeddings(self.ptr)
        cdef vector[float] vec
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
        cdef float * embds = llama_cpp.llama_get_embeddings_ith(self.ptr, i)
        cdef vector[float] vec
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
    #     cdef llama_perf_context_data data = llama_cpp.llama_perf_context(self.ptr)

    def print_perf_data(self):
        """print performance data"""
        llama_cpp.llama_perf_context_print(self.ptr)

    def reset_perf_data(self):
        """reset performance data"""
        llama_cpp.llama_perf_context_reset(self.ptr)



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
    cdef llama_cpp.llama_batch p
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

        self.p = llama_cpp.llama_batch_init(
            self._n_tokens, self.embd, self.n_seq_max
        )

    def __dealloc__(self):
        """The batch has to be freed with `llama_batch_free()`"""
        if self.owner is True:
            llama_cpp.llama_batch_free(self.p)

    def close(self):
        self.__dealloc__()

    @staticmethod
    cdef LlamaBatch from_instance(llama_cpp.llama_batch batch):
        cdef LlamaBatch wrapper = LlamaBatch.__new__(LlamaBatch)
        wrapper.p = batch
        return wrapper

    @property
    def n_tokens(self) -> int:
        return self.p.n_tokens

    def reset(self):
        self.p.n_tokens = 0

    def add(self, llama_cpp.llama_token id, llama_cpp.llama_pos pos, list[int] seq_ids, bint logits):
        cdef vector[llama_cpp.llama_seq_id] _seq_ids
        for i in seq_ids:
            _seq_ids.push_back(i)
        llama_cpp.common_batch_add(self.p, id, pos, _seq_ids, logits)

    def clear(self):
        llama_cpp.common_batch_clear(self.p)

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


cdef class CommonInitResult:
    cdef llama_cpp.common_init_result p
    # cdef llama_cpp.llama_model_ptr model
    # cdef llama_cpp.llama_context_ptr context
    cdef vector[llama_cpp.llama_adapter_lora_ptr] lora

    def __init__(self, params: CommonParams):
        self.p = llama_cpp.common_init_from_params(params.p)

    @property
    def model(self) -> LlamaModel:
        return LlamaModel.from_ptr(self.p.model.get(), owner=True)

    @property
    def context(self) -> LlamaContext:
        return LlamaContext.from_ptr(self.p.context.get(), owner=True)



def ggml_backend_load_all():
    llama_cpp.ggml_backend_load_all()

def common_init():
    """call once at the start of a program if it uses libcommon

    initializes the logging system and prints info about the build
    """
    llama_cpp.common_init()

def llama_backend_init():
    """Initialize the llama + ggml backend

    If numa is true, use NUMA optimizations
    Call once at the start of the program
    """
    llama_cpp.llama_backend_init()

def llama_numa_init(ggml_numa_strategy numa):
    llama_cpp.llama_numa_init(numa)

def common_model_params_to_llama(CommonParams params) -> LlamaModelParams:
    cdef llama_cpp.llama_model_params model_params = llama_cpp.common_model_params_to_llama(params.p)
    return LlamaModelParams.from_instance(model_params)

def common_context_params_to_llama(CommonParams params) -> LlamaContextParams:
    return LlamaContextParams.from_common_params(params)

# def common_tokenize(LlamaContext ctx, str text, bint add_special, bint parse_special = False):
#     return llama_cpp.common_tokenize(<const llama_cpp.llama_context *>ctx.ptr, <string>text.encode(), add_special, parse_special)

# def common_tokenize_from_model(LlamaModel model, str text, bint add_special, bint parse_special = False):
#     return llama_cpp.common_tokenize(<const llama_cpp.llama_model *>model.ptr, <string>text.encode(), add_special, parse_special)

def common_token_to_piece(LlamaContext ctx, int token, bint special = True) -> str:
    return llama_cpp.common_token_to_piece(ctx.ptr, token, special).decode()

def common_batch_add(LlamaBatch batch, llama_cpp.llama_token id, llama_cpp.llama_pos pos, list[int] seq_ids, bint logits):
    return llama_cpp.common_batch_add(batch.p, id, pos, seq_ids, logits)

# def common_chat_format_example(LlamaModel model, str tmpl) -> str:
#     """Returns an example of formatted chat"""
#     return llama_cpp.common_chat_format_example(model.ptr, tmpl.encode()).decode()

def common_params_get_system_info(CommonParams params) -> str:
    return llama_cpp.common_params_get_system_info(params.p).decode()

# def common_chat_format_single(LlamaModel model, tmpl: str, past_msg: list[CommonChatMsg], new_msg: CommonChatMsg, add_ass: bool):
#     """Format single message, while taking into account the position of that message in chat history"""
#     cdef vector[llama_cpp.common_chat_msg] vec
#     for msg in past_msg:
#         vec.push_back(msg.p)
#     return llama_cpp.common_chat_format_single(model.ptr, str.encode(), vec, new_msg.p, add_ass).decode()

def string_from_bool(bint value) -> str:
    return llama_cpp.string_from(value).decode()

def string_from_ints(values: list[int]) -> str:
    cdef vector[int] vec
    for i in values:
        vec.push_back(i)
    return llama_cpp.string_from(vec).decode()

def string_from_tokens(LlamaContext ctx, tokens: list[int]) -> str:
    cdef vector[llama_cpp.llama_token] vec
    for i in tokens:
        vec.push_back(<llama_cpp.llama_token>i)
    return llama_cpp.string_from(ctx.ptr, vec).decode()

def string_from_batch(LlamaContext ctx, LlamaBatch batch) -> str:
    return llama_cpp.string_from(ctx.ptr, batch.p).decode()

def ggml_time_us() -> int:
    return llama_cpp.ggml_time_us()

def set_process_priority(ggml_sched_priority prio) -> bool:
    return llama_cpp.set_process_priority(<ggml_sched_priority> prio)

def llama_time_us() -> int:
    return llama_cpp.llama_time_us()

def llama_max_devices() -> int:
    return llama_cpp.llama_max_devices()

def llama_supports_mmap() -> bool:
    return llama_cpp.llama_supports_mmap()

def llama_supports_mlock() -> bool:
    return llama_cpp.llama_supports_mlock()

def llama_supports_gpu_offload() -> bool:
    return llama_cpp.llama_supports_gpu_offload()

def llama_supports_rpc() -> bool:
    return llama_cpp.llama_supports_rpc()

def llama_attach_threadpool(LlamaContext ctx, GGMLThreadPool threadpool, GGMLThreadPool threadpool_batch):
    llama_cpp.llama_attach_threadpool(ctx.ptr, threadpool.ptr, threadpool_batch.ptr)

def llama_detach_threadpool(LlamaContext ctx):
    llama_cpp.llama_detach_threadpool(ctx.ptr)

def log_set_verbosity(int verbosity):
    llama_cpp.common_log_set_verbosity_thold(verbosity)

def llama_batch_get_one(list[int] tokens) -> LlamaBatch:
    cdef int32_t n_tokens = <int32_t>len(tokens)
    cdef llama_cpp.llama_token * _tokens = <llama_cpp.llama_token *>malloc(sizeof(llama_cpp.llama_token) * n_tokens)
    for i in range(n_tokens):
        _tokens[i] = tokens[i]
    cdef llama_cpp.llama_batch batch = llama_cpp.llama_batch_get_one(_tokens, n_tokens)
    free(_tokens)
    return LlamaBatch.from_instance(batch)

def common_batch_clear(LlamaBatch batch):
    llama_cpp.common_batch_clear(batch.p)

def llama_backend_free():
    """Call once at the end of the program - currently only used for MPI"""
    llama_cpp.llama_backend_free()

# cdef void ggml_abort(const char * file, int line, const char * fmt):
#     llama_cpp.ggml_abort(file, line, fmt)
