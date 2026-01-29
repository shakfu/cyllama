# distutils: language = c++
"""cyllama: a thin cython wrapper of llama.cpp"""


from libc.stdint cimport uint8_t, int32_t, int64_t, uint32_t, uint64_t
from libc.string cimport strlen
from libc.stdlib cimport malloc, calloc, realloc, free
from libcpp.vector cimport vector as std_vector
from libcpp.string cimport string as std_string
from libcpp cimport bool as cppbool  # required for func pointer sigs

# JSON schema to grammar C helper
cdef extern from "helpers/json_schema.h":
    char* json_schema_to_grammar_wrapper(const char* json_schema_str, bint force_gbnf)

cimport ggml
cimport llama
cimport common
cimport sampling
cimport chat
cimport log
cimport gguf
cimport download
cimport ngram_cache
cimport speculative


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
include "mtmd.pxi"
include "speculative.pxi"


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


# Memory Pool System for Performance Optimization
# ==============================================

cdef class TokenMemoryPool:
    """Memory pool for efficient token list reuse

    Reduces frequent small allocations by maintaining pools of
    reusable token lists for common sizes.
    """
    cdef dict _pools          # dict[int, list] - pools by size
    cdef dict _usage_count   # dict[int, int] - usage statistics
    cdef int _max_pool_size  # maximum items per pool
    cdef int _max_token_size # maximum token list size to pool

    def __init__(self, max_pool_size: int = 10, max_token_size: int = 1024):
        """Initialize token memory pool

        Args:
            max_pool_size: Maximum number of lists to cache per size
            max_token_size: Maximum token list size to pool (larger allocations not pooled)
        """
        self._pools = {}
        self._usage_count = {}
        self._max_pool_size = max_pool_size
        self._max_token_size = max_token_size

        # Pre-populate pools for common sizes
        common_sizes = [8, 16, 32, 64, 128, 256, 512]
        for size in common_sizes:
            if size <= max_token_size:
                self._pools[size] = []

    cdef list[int] get_token_list(self, int size):
        """Get a reusable token list of specified size

        Returns either a pooled list or creates a new one if pool is empty.
        """
        cdef list[int] token_list

        # Don't pool very large allocations
        if size > self._max_token_size:
            return [0] * size

        # Track usage for this size
        self._usage_count[size] = self._usage_count.get(size, 0) + 1

        # Try to get from pool
        if size in self._pools and self._pools[size]:
            token_list = self._pools[size].pop()
            # Ensure correct size (lists may have been resized)
            if len(token_list) != size:
                token_list = [0] * size
            else:
                # Clear existing data
                for i in range(size):
                    token_list[i] = 0
            return token_list
        else:
            # Create new list
            return [0] * size

    cdef void return_token_list(self, list[int] token_list):
        """Return a token list to the pool for reuse

        Args:
            token_list: Token list to return to pool
        """
        cdef int size = len(token_list)

        # Don't pool very large allocations
        if size > self._max_token_size:
            return

        # Initialize pool for this size if needed
        if size not in self._pools:
            self._pools[size] = []

        # Only pool if under the limit
        if len(self._pools[size]) < self._max_pool_size:
            self._pools[size].append(token_list)

    def get_stats(self):
        """Get pool usage statistics"""
        stats = {
            "pool_sizes": {size: len(pool) for size, pool in self._pools.items()},
            "usage_count": dict(self._usage_count),
            "total_pools": len(self._pools),
            "total_pooled_lists": sum(len(pool) for pool in self._pools.values())
        }
        return stats

# Global token memory pool instance
cdef TokenMemoryPool _global_token_pool = TokenMemoryPool()

def get_token_pool_stats():
    """Get statistics from the global token memory pool"""
    return _global_token_pool.get_stats()

def reset_token_pool():
    """Reset the global token memory pool (useful for testing)"""
    global _global_token_pool
    _global_token_pool = TokenMemoryPool()

cdef class BatchMemoryPool:
    """Memory pool for efficient LlamaBatch reuse

    Reduces frequent batch object allocations by maintaining pools of
    reusable batch objects for common configurations.
    """
    cdef dict _pools          # dict[tuple, list] - pools by (n_tokens, embd, n_seq_max)
    cdef dict _usage_count   # dict[tuple, int] - usage statistics
    cdef int _max_pool_size  # maximum items per pool
    cdef int _max_batch_size # maximum batch size to pool

    def __init__(self, max_pool_size: int = 5, max_batch_size: int = 512):
        """Initialize batch memory pool

        Args:
            max_pool_size: Maximum number of batches to cache per configuration
            max_batch_size: Maximum batch size to pool (larger allocations not pooled)
        """
        self._pools = {}
        self._usage_count = {}
        self._max_pool_size = max_pool_size
        self._max_batch_size = max_batch_size

    cdef LlamaBatch get_batch(self, int n_tokens, int embd, int n_seq_max):
        """Get a reusable batch of specified configuration

        Returns either a pooled batch or creates a new one if pool is empty.
        """
        cdef tuple key = (n_tokens, embd, n_seq_max)
        cdef LlamaBatch batch

        # Don't pool very large allocations
        if n_tokens > self._max_batch_size:
            return LlamaBatch(n_tokens=n_tokens, embd=embd, n_seq_max=n_seq_max)

        # Track usage for this configuration
        self._usage_count[key] = self._usage_count.get(key, 0) + 1

        # Try to get from pool
        if key in self._pools and self._pools[key]:
            batch = self._pools[key].pop()
            # Reset batch state for reuse
            batch._n_tokens = n_tokens
            batch.p.n_tokens = 0  # Will be set by user
            return batch
        else:
            # Create new batch
            return LlamaBatch(n_tokens=n_tokens, embd=embd, n_seq_max=n_seq_max)

    cdef void return_batch(self, LlamaBatch batch):
        """Return a batch to the pool for reuse

        Args:
            batch: Batch object to return to pool
        """
        cdef tuple key = (batch._n_tokens, batch.embd, batch.n_seq_max)

        # Don't pool very large allocations
        if batch._n_tokens > self._max_batch_size:
            return

        # Initialize pool for this configuration if needed
        if key not in self._pools:
            self._pools[key] = []

        # Only pool if under the limit
        if len(self._pools[key]) < self._max_pool_size:
            self._pools[key].append(batch)

    def get_stats(self):
        """Get pool usage statistics"""
        stats = {
            "pool_configs": {str(config): len(pool) for config, pool in self._pools.items()},
            "usage_count": {str(config): count for config, count in self._usage_count.items()},
            "total_pools": len(self._pools),
            "total_pooled_batches": sum(len(pool) for pool in self._pools.values())
        }
        return stats

# Global batch memory pool instance
cdef BatchMemoryPool _global_batch_pool = BatchMemoryPool()

def get_batch_pool_stats():
    """Get statistics from the global batch memory pool"""
    return _global_batch_pool.get_stats()

def reset_batch_pool():
    """Reset the global batch memory pool (useful for testing)"""
    global _global_batch_pool
    _global_batch_pool = BatchMemoryPool()

def return_batch_to_pool(LlamaBatch batch):
    """Return a batch to the memory pool for reuse

    This should be called when a batch is no longer needed to allow
    it to be reused, reducing memory allocation overhead.

    Args:
        batch: The LlamaBatch object to return to the pool
    """
    _global_batch_pool.return_batch(batch)

def get_pooled_batch(int n_tokens, int embd = 0, int n_seq_max = 1) -> LlamaBatch:
    """Get a batch from the memory pool

    This is an alternative to LlamaBatch() constructor that uses pooling
    for better performance on frequently allocated batch sizes.

    Args:
        n_tokens: Number of tokens in the batch
        embd: Embedding dimension (0 for token mode)
        n_seq_max: Maximum number of sequences

    Returns:
        A LlamaBatch object (either from pool or newly created)
    """
    return _global_batch_pool.get_batch(n_tokens, embd, n_seq_max)

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

    def add(self, llama.llama_token id, llama.llama_pos pos, list[int] seq_ids, bint logits):
        """Add a single token to the batch with position and sequence ID tracking.

        Args:
            id: Token ID to add
            pos: Position in the sequence
            seq_ids: List of sequence IDs this token belongs to
            logits: Whether to compute logits for this token
        """
        cdef std_vector[llama.llama_seq_id] _seq_ids
        for i in seq_ids:
            _seq_ids.push_back(i)
        common.common_batch_add(self.p, id, pos, _seq_ids, logits)

    def clear(self):
        """Clear the batch, resetting n_tokens to 0."""
        common.common_batch_clear(self.p)

    def set_batch(self, batch: Sequence[int], n_past: int, logits_all: bool):
        cdef int n_tokens = len(batch)
        cdef int i
        cdef int past_pos = n_past
        cdef bint logits_flag = logits_all

        self.p.n_tokens = n_tokens

        # Optimized batch setup loop - core operations can run without GIL
        with nogil:
            for i in range(n_tokens):
                self.p.pos[i] = past_pos + i
                self.p.seq_id[i][0] = 0
                self.p.n_seq_id[i] = 1
                self.p.logits[i] = logits_flag

        # Set tokens (requires GIL for Python sequence access)
        for i in range(n_tokens):
            self.p.token[i] = batch[i]

        # Ensure last token generates logits
        self.p.logits[n_tokens - 1] = True

    def add_sequence(self, batch: Sequence[int], seq_id: int, logits_all: bool):
        cdef int n_tokens = len(batch)
        cdef int n_tokens0 = self.p.n_tokens
        cdef int i, j
        cdef int seq_id_val = seq_id
        cdef bint logits_flag = logits_all

        self.p.n_tokens += n_tokens

        # Optimized sequence addition loop - core operations can run without GIL
        with nogil:
            for i in range(n_tokens):
                j = n_tokens0 + i
                self.p.pos[j] = i
                self.p.seq_id[j][0] = seq_id_val
                self.p.n_seq_id[j] = 1
                self.p.logits[j] = logits_flag

        # Set tokens (requires GIL for Python sequence access)
        for i in range(n_tokens):
            j = n_tokens0 + i
            self.p.token[j] = batch[i]

        # Ensure last token generates logits
        self.p.logits[n_tokens0 + n_tokens - 1] = True

    def set_last_logits_to_true(self):
        # Simple operation can run without GIL
        with nogil:
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
    cdef object _progress_callback  # prevent garbage collection of Python callback
    cdef float * _tensor_split      # owned buffer for tensor_split values

    def __init__(self):
        self.p = llama.llama_model_default_params()
        self._progress_callback = None
        self._tensor_split = NULL

    def __dealloc__(self):
        if self._tensor_split != NULL:
            free(self._tensor_split)
            self._tensor_split = NULL

    @staticmethod
    cdef LlamaModelParams from_instance(llama.llama_model_params params):
        cdef LlamaModelParams wrapper = LlamaModelParams.__new__(LlamaModelParams)
        wrapper.p = params
        wrapper._progress_callback = None
        wrapper._tensor_split = NULL
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
        """Proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()

        Each value represents the proportion of work to assign to each GPU.
        Values are normalized automatically by llama.cpp, so [1, 1] and [0.5, 0.5]
        are equivalent (50% each). Use [1, 2] to assign 1/3 to GPU 0 and 2/3 to GPU 1.

        Example:
            params = LlamaModelParams()
            params.split_mode = 1  # LLAMA_SPLIT_MODE_LAYER
            params.tensor_split = [0.5, 0.5]  # Split equally between 2 GPUs
        """
        cdef size_t length = llama.llama_max_devices()
        results = []
        if self.p.tensor_split:
            for i in range(length):
                n = <float>self.p.tensor_split[i]
                results.append(n)
        return results

    @tensor_split.setter
    def tensor_split(self, values):
        """Set tensor split proportions for multi-GPU inference.

        Args:
            values: List of floats representing proportions for each GPU.
                    Length should not exceed llama_max_devices().
                    Pass None or empty list to clear (use default distribution).
        """
        cdef size_t max_devices = llama.llama_max_devices()
        cdef size_t i

        # Handle None or empty list - clear the tensor_split
        if values is None or len(values) == 0:
            if self._tensor_split != NULL:
                free(self._tensor_split)
                self._tensor_split = NULL
            self.p.tensor_split = NULL
            return

        if len(values) > max_devices:
            raise ValueError(f"tensor_split has {len(values)} elements but max devices is {max_devices}")

        # Allocate buffer if not already allocated
        if self._tensor_split == NULL:
            self._tensor_split = <float *>calloc(max_devices, sizeof(float))
            if self._tensor_split == NULL:
                raise MemoryError("Failed to allocate tensor_split buffer")

        # Copy values to buffer, zero-fill remaining slots
        for i in range(max_devices):
            if i < len(values):
                self._tensor_split[i] = <float>values[i]
            else:
                self._tensor_split[i] = 0.0

        # Point the params to our buffer
        self.p.tensor_split = self._tensor_split

    @property
    def progress_callback(self) -> Callable[[float], bool]:
        """Called with a progress value between 0.0 and 1.0. Pass None to disable.

        If the provided progress_callback returns true, model loading continues.
        If it returns false, model loading is immediately aborted.

        Example:
            def on_progress(progress: float) -> bool:
                print(f"Loading: {progress * 100:.1f}%")
                return True  # continue loading

            params = LlamaModelParams()
            params.progress_callback = on_progress
        """
        return self._progress_callback

    @progress_callback.setter
    def progress_callback(self, object py_progress_callback):
        if py_progress_callback is None:
            self._progress_callback = None
            self.p.progress_callback = NULL
            self.p.progress_callback_user_data = NULL
        else:
            self._progress_callback = py_progress_callback
            self.p.progress_callback = <llama.llama_progress_callback>&progress_callback
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
    def use_direct_io(self) -> bool:
        """Use direct I/O, takes precedence over use_mmap"""
        return self.p.use_direct_io

    @use_direct_io.setter
    def use_direct_io(self, value: bool):
        self.p.use_direct_io = value

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

    @property
    def use_extra_bufts(self) -> bool:
        """Use extra buffer types (used for weight repacking)"""
        return self.p.use_extra_bufts

    @use_extra_bufts.setter
    def use_extra_bufts(self, value: bool):
        self.p.use_extra_bufts = value

    @property
    def no_host(self) -> bool:
        """Bypass host buffer allowing extra buffers to be used"""
        return self.p.no_host

    @no_host.setter
    def no_host(self, value: bool):
        self.p.no_host = value

    @property
    def no_alloc(self) -> bool:
        """Only load metadata and simulate memory allocations"""
        return self.p.no_alloc

    @no_alloc.setter
    def no_alloc(self, value: bool):
        self.p.no_alloc = value


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

        # OLD WAY
        # Pre-allocate result list with known size for better performance
        # cdef list[int] _tokens = [0] * n_tokens  # Pre-allocate with correct size

        # Get token list from memory pool for better performance
        cdef list[int] _tokens = _global_token_pool.get_token_list(n_tokens)

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
    cdef int _cached_n_embd_inp
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
        self._cached_n_embd_inp = -1
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
            self._cached_n_embd_inp = llama.llama_model_n_embd_inp(self.ptr)
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
    def n_embd_inp(self) -> int:
        if self._cache_initialized:
            return self._cached_n_embd_inp
        return llama.llama_model_n_embd_inp(self.ptr)

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
        """Load a LoRA adapter from file.

        The loaded adapter will be associated to the given model, and will be freed when the model is deleted.

        Args:
            path_lora: Path to the LoRA adapter file.

        Returns:
            LlamaAdapterLora: The loaded LoRA adapter.

        Raises:
            FileNotFoundError: If the LoRA adapter file does not exist.
            ValueError: If loading the LoRA adapter fails.
        """
        if not os.path.exists(path_lora):
            raise FileNotFoundError(f"LoRA adapter file not found: {path_lora}")

        cdef llama.llama_adapter_lora * ptr = llama.llama_adapter_lora_init(
            self.ptr, path_lora.encode())
        if ptr is NULL:
            raise ValueError(f"Failed to load LoRA adapter from: {path_lora}")
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

    def is_hybrid(self) -> bool:
        """Returns true if the model is hybrid (like Jamba, Granite, etc.)"""
        return llama.llama_model_is_hybrid(self.ptr)

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
        if model is None:
            raise ValueError("model cannot be None")
        if not isinstance(model, LlamaModel):
            raise TypeError(f"model must be LlamaModel, got {type(model).__name__}")
        if model.ptr is NULL:
            raise ValueError("model has been freed or is invalid (NULL pointer)")

        self.model = model
        self.params = params if params else LlamaContextParams()
        self.verbose = verbose

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
    def n_ctx_seq(self) -> int:
        return llama.llama_n_ctx_seq(self.ptr)

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
        """Load session file.

        Args:
            path_session: Path to the session state file.
            max_n_tokens: Maximum number of tokens to load.

        Returns:
            List of tokens from the session file.

        Raises:
            FileNotFoundError: If the session file does not exist.
        """
        if not os.path.exists(path_session):
            raise FileNotFoundError(f"Session state file not found: {path_session}")

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
        """Save session file.

        Args:
            path_session: Path where the session state file will be saved.
            tokens: List of tokens to save.

        Returns:
            True if save was successful.

        Raises:
            FileNotFoundError: If the parent directory does not exist.
        """
        parent_dir = os.path.dirname(path_session)
        if parent_dir and not os.path.exists(parent_dir):
            raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

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
        """Copy the KV cache of a single sequence into a dynamically allocated buffer.

        Returns the sequence data as a list of bytes.
        """
        # Get the required size first to avoid buffer overflow
        cdef size_t required_size = llama.llama_state_seq_get_size(self.ptr, seq_id)
        cdef uint8_t * dst = NULL
        cdef size_t copied = 0
        cdef std_vector[uint8_t] result

        if required_size == 0:
            return []

        # Dynamically allocate buffer of exact size needed
        dst = <uint8_t *>malloc(required_size)
        if dst is NULL:
            raise MemoryError("Failed to allocate buffer for state sequence data")

        try:
            copied = llama.llama_state_seq_get_data(
                self.ptr, dst, required_size, seq_id)
            for i in range(copied):
                result.push_back(dst[i])
            return result
        finally:
            free(dst)

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
        """Save state sequence data to a file.

        Args:
            filepath: Path where the sequence state file will be saved.
            seq_id: Sequence ID to save.
            tokens: List of tokens to save.

        Raises:
            FileNotFoundError: If the parent directory does not exist.
            ValueError: If saving fails.
        """
        parent_dir = os.path.dirname(filepath)
        if parent_dir and not os.path.exists(parent_dir):
            raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

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
        """Load state sequence data from a file.

        Args:
            filepath: Path to the sequence state file.
            dest_seq_id: Destination sequence ID.
            max_n_tokens: Maximum number of tokens to load.

        Returns:
            List of tokens from the sequence file.

        Raises:
            FileNotFoundError: If the sequence state file does not exist.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Sequence state file not found: {filepath}")

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
        """Get state sequence data from seq_id and flags using dynamically allocated buffer.

        Returns the sequence data as a list of bytes.
        """
        # Get the required size first to avoid buffer overflow
        cdef size_t required_size = llama.llama_state_seq_get_size_ext(self.ptr, seq_id, flags)
        cdef uint8_t * dst = NULL
        cdef size_t size = 0
        cdef std_vector[uint8_t] result

        if required_size == 0:
            return []

        # Dynamically allocate buffer of exact size needed
        dst = <uint8_t *>malloc(required_size)
        if dst is NULL:
            raise MemoryError("Failed to allocate buffer for state sequence data")

        try:
            size = llama.llama_state_seq_get_data_ext(self.ptr, dst, required_size, seq_id, flags)
            for i in range(size):
                result.push_back(dst[i])
            return result
        finally:
            free(dst)

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
        # Optimized decode operation with minimal Python overhead
        cdef int32_t res = llama.llama_decode(self.ptr, batch.p)
        cdef int32_t batch_tokens = batch.n_tokens

        self.n_tokens = batch_tokens

        # Fast error checking with pre-computed conditions
        if res == 1:
            raise ValueError("could not find a KV slot for the batch (try reducing the size of the batch or increase the context)")
        elif res < 0:
            raise RuntimeError("llama_decode failed")

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

    # Memory / KV Cache Management
    # -------------------------------------------------------------------------

    def kv_cache_clear(self, bint clear_data=True):
        """Clear the KV cache.

        This removes all cached key-value pairs from the context's memory,
        allowing the context to be reused for new generations without
        recreating it.

        Args:
            clear_data: If True (default), also clear the data buffers.
                       If False, only clear metadata.

        Note:
            This is useful for reusing a context across multiple independent
            generations without the overhead of context recreation.
        """
        cdef llama.llama_memory_t mem = llama.llama_get_memory(self.ptr)
        if mem is not NULL:
            llama.llama_memory_clear(mem, clear_data)

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
        cdef LlamaSamplerChainParams _params
        if not params:
            _params = LlamaSamplerChainParams()
        else:
            _params = params
        self.params = _params
        self.ptr = llama.llama_sampler_chain_init(_params.p)

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

    def add_mirostat(self, int n_vocab, uint32_t seed, float tau, float eta, int m):
        """Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966.

        Uses tokens instead of words.

        Args:
            n_vocab: Size of the vocabulary.
            seed: Random seed for sampling.
            tau: The target cross-entropy (or surprise) value. A higher value
                corresponds to more surprising or less predictable text.
            eta: The learning rate used to update `mu` based on the error between
                the target and observed surprisal. Larger values update faster.
            m: The number of tokens considered in the estimation of `s_hat`.
                The paper uses m=100, but other values can be experimented with.
        """
        llama.llama_sampler_chain_add(
            self.ptr, llama.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m))

    def add_mirostat_v2(self, uint32_t seed, float tau, float eta):
        """Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966.

        Uses tokens instead of words. This is a simplified version of Mirostat
        that doesn't require the vocabulary size or m parameter.

        Args:
            seed: Random seed for sampling.
            tau: The target cross-entropy (or surprise) value. A higher value
                corresponds to more surprising or less predictable text.
            eta: The learning rate used to update `mu` based on the error between
                the target and observed surprisal. Larger values update faster.
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

    def add_logit_bias(self, int n_vocab, logit_biases: list):
        """Add logit bias sampler to modify token probabilities.

        Applies additive biases to specific token logits before sampling.
        Positive bias increases probability, negative decreases it.

        Args:
            n_vocab: Size of the vocabulary.
            logit_biases: List of (token_id, bias) tuples. Each tuple contains
                a token ID (int) and a bias value (float) to add to that token's logit.

        Example:
            # Increase probability of token 123, decrease token 456
            sampler.add_logit_bias(vocab_size, [(123, 5.0), (456, -5.0)])
        """
        cdef int n_logit_bias = len(logit_biases)
        cdef llama.llama_logit_bias* bias_array = NULL

        if n_logit_bias > 0:
            bias_array = <llama.llama_logit_bias*>malloc(
                n_logit_bias * sizeof(llama.llama_logit_bias))
            if bias_array == NULL:
                raise MemoryError("Failed to allocate logit bias array")

            try:
                for i, (token, bias) in enumerate(logit_biases):
                    bias_array[i].token = token
                    bias_array[i].bias = bias

                llama.llama_sampler_chain_add(
                    self.ptr, llama.llama_sampler_init_logit_bias(
                        n_vocab, n_logit_bias, bias_array))
            finally:
                free(bias_array)
        else:
            llama.llama_sampler_chain_add(
                self.ptr, llama.llama_sampler_init_logit_bias(n_vocab, 0, NULL))

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
        # Optimized sampling with minimal Python overhead
        cdef llama.llama_token result = llama.llama_sampler_sample(self.ptr, ctx.ptr, idx)
        return result


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
    """Create a batch using the proper batch API with optimized token handling and memory pooling"""
    cdef int32_t n_tokens = <int32_t>len(tokens)
    cdef int i

    # OLD WAY
    # Create a proper batch using the new API
    # batch = LlamaBatch(n_tokens=n_tokens, embd=0, n_seq_max=1)

    # Get batch from memory pool for better performance
    batch = _global_batch_pool.get_batch(n_tokens, 0, 1)

    # Optimized batch creation - set up structure efficiently
    batch.p.n_tokens = n_tokens

    # Fast setup of positions, sequence IDs, and logits without GIL
    with nogil:
        for i in range(n_tokens):
            batch.p.pos[i] = n_past + i
            batch.p.seq_id[i][0] = 0
            batch.p.n_seq_id[i] = 1
            batch.p.logits[i] = False  # Default to False for all tokens
        # Last token should generate logits
        batch.p.logits[n_tokens - 1] = True

    # Set tokens (requires GIL for Python list access)
    for i in range(n_tokens):
        batch.p.token[i] = tokens[i]

    return batch

def llama_backend_free():
    """Call once at the end of the program - currently only used for MPI"""
    llama.llama_backend_free()

def llama_flash_attn_type_name(llama.llama_flash_attn_type flash_attn_type) -> str:
    return llama.llama_flash_attn_type_name(flash_attn_type).decode()


#------------------------------------------------------------------------------
# GGUF File Format API
#------------------------------------------------------------------------------

cdef class GGUFContext:
    """
    Wrapper for GGUF file format context.

    GGUF (GGML Universal File Format) is the binary file format used by ggml
    for storing models. It contains:
    - Key-value metadata
    - Tensor information (names, shapes, types, offsets)
    - Tensor data

    Example usage:
        # Read GGUF file
        ctx = GGUFContext.from_file("model.gguf")
        print(f"Version: {ctx.version}")
        print(f"Tensors: {ctx.n_tensors}")
        print(f"Metadata: {ctx.get_all_metadata()}")

        # Create new GGUF
        ctx = GGUFContext.empty()
        ctx.set_val_str("model.name", "MyModel")
        ctx.set_val_u32("model.version", 1)
        ctx.write_to_file("output.gguf")
    """
    cdef gguf.gguf_context * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = NULL
        self.owner = False

    def __dealloc__(self):
        if self.ptr != NULL and self.owner:
            gguf.gguf_free(self.ptr)
            self.ptr = NULL

    @staticmethod
    def empty():
        """Create an empty GGUF context."""
        cdef GGUFContext ctx = GGUFContext.__new__(GGUFContext)
        ctx.ptr = gguf.gguf_init_empty()
        if ctx.ptr == NULL:
            raise MemoryError("Failed to create empty GGUF context")
        ctx.owner = True
        return ctx

    @staticmethod
    def from_file(str filename, bint no_alloc=True):
        """
        Load GGUF context from file.

        Args:
            filename: Path to GGUF file
            no_alloc: If True, don't allocate tensor data in memory

        Returns:
            GGUFContext object
        """
        cdef GGUFContext ctx = GGUFContext.__new__(GGUFContext)
        cdef gguf.gguf_init_params params
        params.no_alloc = no_alloc
        params.ctx = NULL

        filename_bytes = filename.encode('utf-8')
        ctx.ptr = gguf.gguf_init_from_file(filename_bytes, params)
        if ctx.ptr == NULL:
            raise IOError(f"Failed to load GGUF file: {filename}")
        ctx.owner = True
        return ctx

    @property
    def version(self) -> int:
        """Get GGUF file version."""
        return gguf.gguf_get_version(self.ptr)

    @property
    def alignment(self) -> int:
        """Get tensor data alignment."""
        return gguf.gguf_get_alignment(self.ptr)

    @property
    def data_offset(self) -> int:
        """Get offset to tensor data in file."""
        return gguf.gguf_get_data_offset(self.ptr)

    @property
    def n_kv(self) -> int:
        """Get number of key-value pairs."""
        return gguf.gguf_get_n_kv(self.ptr)

    @property
    def n_tensors(self) -> int:
        """Get number of tensors."""
        return gguf.gguf_get_n_tensors(self.ptr)

    def find_key(self, str key) -> int:
        """
        Find key by name.

        Args:
            key: Key name to search for

        Returns:
            Key ID (>= 0) if found, -1 if not found
        """
        key_bytes = key.encode('utf-8')
        return gguf.gguf_find_key(self.ptr, key_bytes)

    def get_key(self, int key_id) -> str:
        """Get key name by ID."""
        cdef const char * key_c = gguf.gguf_get_key(self.ptr, key_id)
        if key_c == NULL:
            raise ValueError(f"Invalid key ID: {key_id}")
        return key_c.decode('utf-8')

    def get_kv_type(self, int key_id) -> int:
        """Get value type for key."""
        return gguf.gguf_get_kv_type(self.ptr, key_id)

    def get_value(self, str key):
        """
        Get value by key name (auto-detects type).

        Args:
            key: Key name

        Returns:
            Value (type depends on GGUF type)
        """
        cdef int64_t key_id = self.find_key(key)
        if key_id < 0:
            raise KeyError(f"Key not found: {key}")

        cdef int vtype = self.get_kv_type(key_id)
        cdef const char * s

        if vtype == gguf.GGUF_TYPE_UINT8:
            return gguf.gguf_get_val_u8(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_INT8:
            return gguf.gguf_get_val_i8(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_UINT16:
            return gguf.gguf_get_val_u16(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_INT16:
            return gguf.gguf_get_val_i16(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_UINT32:
            return gguf.gguf_get_val_u32(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_INT32:
            return gguf.gguf_get_val_i32(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_FLOAT32:
            return gguf.gguf_get_val_f32(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_UINT64:
            return gguf.gguf_get_val_u64(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_INT64:
            return gguf.gguf_get_val_i64(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_FLOAT64:
            return gguf.gguf_get_val_f64(self.ptr, key_id)
        elif vtype == gguf.GGUF_TYPE_BOOL:
            return bool(gguf.gguf_get_val_bool(self.ptr, key_id))
        elif vtype == gguf.GGUF_TYPE_STRING:
            s = gguf.gguf_get_val_str(self.ptr, key_id)
            return s.decode('utf-8') if s != NULL else None
        elif vtype == gguf.GGUF_TYPE_ARRAY:
            return self._get_array_value(key_id)
        else:
            raise ValueError(f"Unknown GGUF type: {vtype}")

    def _get_array_value(self, int64_t key_id):
        """Get array value by key ID."""
        cdef int arr_type = gguf.gguf_get_arr_type(self.ptr, key_id)
        cdef size_t n = gguf.gguf_get_arr_n(self.ptr, key_id)
        cdef const char * s
        cdef size_t i

        if arr_type == gguf.GGUF_TYPE_STRING:
            result = []
            for i in range(n):
                s = gguf.gguf_get_arr_str(self.ptr, key_id, i)
                result.append(s.decode('utf-8') if s != NULL else None)
            return result
        else:
            # For numeric arrays, return raw pointer info
            # User would need to use get_arr_data_raw for direct access
            return {"type": arr_type, "length": n}

    def get_all_metadata(self) -> dict:
        """
        Get all key-value metadata as a dictionary.

        Returns:
            Dictionary of all metadata
        """
        result = {}
        cdef int64_t n = self.n_kv
        for i in range(n):
            key = self.get_key(i)
            try:
                result[key] = self.get_value(key)
            except Exception as e:
                # Skip keys that can't be read
                result[key] = f"<error: {e}>"
        return result

    def set_val_str(self, str key, str value):
        """Set string value."""
        key_bytes = key.encode('utf-8')
        value_bytes = value.encode('utf-8')
        gguf.gguf_set_val_str(self.ptr, key_bytes, value_bytes)

    def set_val_bool(self, str key, bint value):
        """Set boolean value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_bool(self.ptr, key_bytes, value)

    def set_val_u8(self, str key, int value):
        """Set uint8 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_u8(self.ptr, key_bytes, value)

    def set_val_i8(self, str key, int value):
        """Set int8 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_i8(self.ptr, key_bytes, value)

    def set_val_u16(self, str key, int value):
        """Set uint16 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_u16(self.ptr, key_bytes, value)

    def set_val_i16(self, str key, int value):
        """Set int16 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_i16(self.ptr, key_bytes, value)

    def set_val_u32(self, str key, int value):
        """Set uint32 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_u32(self.ptr, key_bytes, value)

    def set_val_i32(self, str key, int value):
        """Set int32 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_i32(self.ptr, key_bytes, value)

    def set_val_f32(self, str key, float value):
        """Set float32 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_f32(self.ptr, key_bytes, value)

    def set_val_u64(self, str key, int value):
        """Set uint64 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_u64(self.ptr, key_bytes, value)

    def set_val_i64(self, str key, int value):
        """Set int64 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_i64(self.ptr, key_bytes, value)

    def set_val_f64(self, str key, float value):
        """Set float64 value."""
        key_bytes = key.encode('utf-8')
        gguf.gguf_set_val_f64(self.ptr, key_bytes, value)

    def remove_key(self, str key) -> int:
        """
        Remove key if it exists.

        Args:
            key: Key name to remove

        Returns:
            Key ID that was removed, or -1 if key didn't exist
        """
        key_bytes = key.encode('utf-8')
        return gguf.gguf_remove_key(self.ptr, key_bytes)

    def find_tensor(self, str name) -> int:
        """
        Find tensor by name.

        Args:
            name: Tensor name

        Returns:
            Tensor ID (>= 0) if found, -1 if not found
        """
        name_bytes = name.encode('utf-8')
        return gguf.gguf_find_tensor(self.ptr, name_bytes)

    def get_tensor_name(self, int tensor_id) -> str:
        """Get tensor name by ID."""
        cdef const char * name = gguf.gguf_get_tensor_name(self.ptr, tensor_id)
        if name == NULL:
            raise ValueError(f"Invalid tensor ID: {tensor_id}")
        return name.decode('utf-8')

    def get_tensor_type(self, int tensor_id) -> int:
        """Get tensor type (ggml_type enum)."""
        return gguf.gguf_get_tensor_type(self.ptr, tensor_id)

    def get_tensor_offset(self, int tensor_id) -> int:
        """Get tensor data offset in file."""
        return gguf.gguf_get_tensor_offset(self.ptr, tensor_id)

    def get_tensor_size(self, int tensor_id) -> int:
        """Get tensor size in bytes."""
        return gguf.gguf_get_tensor_size(self.ptr, tensor_id)

    def get_all_tensor_info(self) -> list:
        """
        Get information about all tensors.

        Returns:
            List of dicts with tensor info
        """
        result = []
        cdef int64_t n = self.n_tensors
        for i in range(n):
            result.append({
                "id": i,
                "name": self.get_tensor_name(i),
                "type": self.get_tensor_type(i),
                "offset": self.get_tensor_offset(i),
                "size": self.get_tensor_size(i),
            })
        return result

    def write_to_file(self, str filename, bint only_meta=False) -> bool:
        """
        Write GGUF context to file.

        Args:
            filename: Output file path
            only_meta: If True, write only metadata (no tensor data)

        Returns:
            True if successful
        """
        filename_bytes = filename.encode('utf-8')
        return gguf.gguf_write_to_file(self.ptr, filename_bytes, only_meta)

    def get_meta_size(self) -> int:
        """Get size of metadata in bytes."""
        return gguf.gguf_get_meta_size(self.ptr)

    def __repr__(self):
        return f"<GGUFContext: version={self.version}, tensors={self.n_tensors}, kv_pairs={self.n_kv}>"


#------------------------------------------------------------------------------
# JSON Schema to Grammar API
#------------------------------------------------------------------------------

def json_schema_to_grammar(schema, force_gbnf=False):
    """
    Convert a JSON schema to GBNF grammar for constrained generation.

    This allows you to generate JSON output that conforms to a specific schema.

    Args:
        schema: JSON schema as dict or JSON string
        force_gbnf: Force GBNF format (default: False)

    Returns:
        str: GBNF grammar string that can be used with llama.cpp

    Raises:
        ValueError: If schema is invalid or cannot be converted

    Example:
        ```python
        # Define a schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"}
            },
            "required": ["name", "age"]
        }

        # Convert to grammar
        grammar = json_schema_to_grammar(schema)

        # Use with model
        result = model.generate(
            prompt="Extract person info: John is 30 years old, email: john@example.com",
            grammar=grammar
        )
        # result will be valid JSON matching the schema
        ```
    """
    import json

    # Convert schema to JSON string if it's a dict
    if isinstance(schema, dict):
        schema_str = json.dumps(schema)
    elif isinstance(schema, str):
        # Validate it's valid JSON
        try:
            json.loads(schema)
            schema_str = schema
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON schema string: {e}")
    else:
        raise TypeError(f"Schema must be dict or str, got {type(schema)}")

    # Convert to bytes for C function
    cdef bytes schema_bytes = schema_str.encode('utf-8')
    cdef const char* schema_c = schema_bytes

    # Call C wrapper
    cdef char* grammar_c = json_schema_to_grammar_wrapper(schema_c, force_gbnf)

    if grammar_c == NULL:
        raise ValueError("Failed to convert JSON schema to grammar. Check that the schema is valid.")

    try:
        # Convert C string to Python string
        grammar_str = grammar_c.decode('utf-8')
        return grammar_str
    finally:
        # Free the C string
        free(grammar_c)


# =============================================================================
# Download API
# =============================================================================

def get_hf_file(hf_repo_with_tag, bearer_token="", offline=False):
    """
    Get HF file information from HuggingFace repo with optional tag.

    Supports Ollama-style tags:
    - bartowski/Llama-3.2-3B-Instruct-GGUF:q4
    - bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
    - bartowski/Llama-3.2-3B-Instruct-GGUF:q5_k_s

    Tag is optional, defaults to "latest" (checks Q4_K_M, then Q4, then first GGUF).

    Args:
        hf_repo_with_tag: HuggingFace repo with optional :tag suffix
        bearer_token: HuggingFace API token (optional)
        offline: If True, only check local cache

    Returns:
        dict with keys: repo, gguf_file, mmproj_file

    Raises:
        RuntimeError: If download/manifest fetch fails

    Example:
        info = get_hf_file("bartowski/Llama-3.2-3B-Instruct-GGUF:q4")
        print(info['gguf_file'])
    """
    cdef bytes repo_bytes = hf_repo_with_tag.encode('utf-8')
    cdef bytes token_bytes = bearer_token.encode('utf-8')

    cdef download.common_hf_file_res res = download.common_get_hf_file(
        repo_bytes,
        token_bytes,
        offline
    )

    return {
        'repo': res.repo.decode('utf-8'),
        'gguf_file': res.ggufFile.decode('utf-8'),
        'mmproj_file': res.mmprojFile.decode('utf-8')
    }


def download_model(model_path=None, url=None, hf_repo=None, hf_file=None,
                   docker_repo=None, bearer_token="", offline=False):
    """
    Download a model from various sources.

    Args:
        model_path: Local path to save model, OR HuggingFace repo shorthand
                   (e.g., "user/repo:tag" is auto-detected as hf_repo)
        url: Direct URL to download from (optional)
        hf_repo: HuggingFace repo name (optional, can include :tag like ":latest")
        hf_file: Specific file in HF repo (optional, auto-resolved if not provided)
        docker_repo: Docker registry repo (optional)
        bearer_token: HuggingFace API token (optional)
        offline: If True, only use local cache

    Returns:
        True if download succeeded, False otherwise

    Examples:
        # Shorthand: auto-detects HuggingFace repo format
        download_model("bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

        # Explicit HuggingFace repo
        download_model(hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF:latest")

        # Download specific file
        download_model(
            hf_repo="bartowski/Llama-3.2-1B-Instruct-GGUF",
            hf_file="Llama-3.2-1B-Instruct-Q4_K_M.gguf"
        )

        # Download from URL
        download_model(url="https://example.com/model.gguf")

        # Download from Docker registry
        download_model(docker_repo="registry.example.com/model:tag")
    """
    import os

    # Auto-detect HuggingFace repo format in model_path
    # Pattern: "user/repo" or "user/repo:tag" (not a local path, not a URL)
    if model_path and hf_repo is None and url is None:
        path_str = str(model_path)
        # Check if it looks like a HF repo: "user/repo" or "user/repo:tag"
        # Must have exactly one /, not be a URL, not be an absolute path,
        # and not contain backslashes (Windows paths)
        is_hf = ('/' in path_str and
            path_str.count('/') == 1 and
            not path_str.startswith(('http://', 'https://', 'file://', '/')) and
            '\\' not in path_str and
            not os.path.exists(path_str))
        if is_hf:
            # Looks like "user/repo" or "user/repo:tag"
            hf_repo = model_path
            model_path = None

    # Handle HuggingFace repo - resolve to URL
    if hf_repo:
        # Get file info from HF API
        hf_info = get_hf_file(hf_repo, bearer_token, offline)
        repo = hf_info['repo']  # repo with tag removed
        gguf_file = hf_file or hf_info['gguf_file']

        if not gguf_file:
            raise ValueError(f"Could not determine GGUF file for repo: {hf_repo}")

        # Construct HuggingFace download URL
        url = f"https://huggingface.co/{repo}/resolve/main/{gguf_file}"

        # Set default local path if not specified
        if not model_path:
            # Use cache directory structure
            cache_dir = os.path.expanduser("~/.cache/llama.cpp")
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(cache_dir, gguf_file)

    cdef download.common_params_model model

    # Set model parameters
    if model_path:
        model.path = model_path.encode('utf-8')
    if url:
        model.url = url.encode('utf-8')
    if docker_repo:
        model.docker_repo = docker_repo.encode('utf-8')

    cdef bytes token_bytes = bearer_token.encode('utf-8')

    cdef bint success = download.common_download_model(model, token_bytes, offline)

    return success


def list_cached_models():
    """
    List all models in the local cache.

    Returns:
        List of dicts with keys: manifest_path, user, model, tag, size

    Example:
        models = list_cached_models()
        for m in models:
            print(f"{m['user']}/{m['model']}:{m['tag']} - {m['size']} bytes")
    """
    cdef std_vector[download.common_cached_model_info] cached = download.common_list_cached_models()

    result = []
    cdef size_t i
    for i in range(cached.size()):
        result.append({
            'manifest_path': cached[i].manifest_path.decode('utf-8'),
            'user': cached[i].user.decode('utf-8'),
            'model': cached[i].model.decode('utf-8'),
            'tag': cached[i].tag.decode('utf-8'),
            'size': cached[i].size
        })

    return result


def resolve_docker_model(docker_repo):
    """
    Resolve and download model from Docker registry.

    Args:
        docker_repo: Docker registry repository (e.g., "registry.example.com/model:tag")

    Returns:
        Local path to downloaded model file

    Raises:
        RuntimeError: If Docker resolution/download fails

    Example:
        path = resolve_docker_model("myregistry.io/llama:latest")
        model = LlamaModel(path)
    """
    cdef bytes repo_bytes = docker_repo.encode('utf-8')
    cdef std_string path = download.common_docker_resolve_model(repo_bytes)
    return path.decode('utf-8')


# =============================================================================
# N-gram Cache API
# =============================================================================

cdef class NgramCache:
    """
    N-gram cache for accelerating text generation with repeated patterns.

    N-gram caching stores patterns of previously generated tokens and uses them
    to predict likely continuations, speeding up generation when text contains
    repetitive patterns.

    Example:
        # Create and update cache
        cache = NgramCache()
        tokens = [1, 2, 3, 4, 5, 2, 3, 4]  # Repeated pattern [2,3,4]
        cache.update(tokens, ngram_min=2, ngram_max=4)

        # Use for drafting
        inp = [1, 2]
        draft = cache.draft(inp, n_draft=5, ngram_min=2, ngram_max=4)

        # Save/load
        cache.save("cache.bin")
        cache2 = NgramCache.load("cache.bin")

        # Merge caches
        cache.merge(cache2)
    """
    cdef ngram_cache.common_ngram_cache * ptr
    cdef bint owner

    def __cinit__(self):
        self.ptr = new ngram_cache.common_ngram_cache()
        self.owner = True

    def __dealloc__(self):
        if self.owner and self.ptr != NULL:
            del self.ptr

    def update(self, tokens, ngram_min=2, ngram_max=4, nnew=None, print_progress=False):
        """
        Update the n-gram cache with new tokens.

        Args:
            tokens: List of token IDs to add to the cache
            ngram_min: Minimum n-gram size (default: 2)
            ngram_max: Maximum n-gram size (default: 4, max: LLAMA_NGRAM_MAX)
            nnew: Number of new tokens appended (default: len(tokens))
            print_progress: Print progress to stderr (default: False)

        Note:
            For correct results, tokens should ONLY be appended to.
            Changes in the middle require rebuilding the cache.

        Example:
            cache = NgramCache()
            cache.update([1, 2, 3, 4, 5])
            cache.update([1, 2, 3, 4, 5, 6, 7], nnew=2)  # Only 6, 7 are new
        """
        cdef std_vector[llama.llama_token] token_vec
        cdef int i

        # Convert Python list to C++ vector
        for token in tokens:
            token_vec.push_back(token)

        # Default nnew to all tokens if not specified
        if nnew is None:
            nnew = len(tokens)

        # Clamp to valid range
        ngram_min = max(ngram_cache.LLAMA_NGRAM_MIN, min(ngram_min, ngram_cache.LLAMA_NGRAM_MAX))
        ngram_max = max(ngram_cache.LLAMA_NGRAM_MIN, min(ngram_max, ngram_cache.LLAMA_NGRAM_MAX))

        ngram_cache.common_ngram_cache_update(
            self.ptr[0],
            ngram_min,
            ngram_max,
            token_vec,
            nnew,
            print_progress
        )

    def draft(self, inp, n_draft=16, ngram_min=2, ngram_max=4,
              context_cache=None, dynamic_cache=None, static_cache=None):
        """
        Draft tokens using n-gram prediction.

        Args:
            inp: Input tokens generated so far
            n_draft: Maximum number of tokens to draft (default: 16)
            ngram_min: Minimum n-gram size (default: 2)
            ngram_max: Maximum n-gram size (default: 4)
            context_cache: NgramCache based on current context (default: self)
            dynamic_cache: NgramCache based on previous generations (default: empty)
            static_cache: NgramCache from large corpus for validation (default: empty)

        Returns:
            List of drafted token IDs

        Example:
            cache = NgramCache()
            cache.update([1, 2, 3, 4, 5, 2, 3])

            inp = [1, 2]
            draft = cache.draft(inp, n_draft=5)
            # Likely to predict [3, 4, ...] based on pattern
        """
        cdef std_vector[llama.llama_token] inp_vec
        cdef std_vector[llama.llama_token] draft_vec
        cdef NgramCache ctx_cache = context_cache if context_cache is not None else self
        cdef NgramCache dyn_cache = dynamic_cache if dynamic_cache is not None else NgramCache()
        cdef NgramCache sta_cache = static_cache if static_cache is not None else NgramCache()
        cdef int i

        # Convert input to C++ vector
        for token in inp:
            inp_vec.push_back(token)

        # Initialize draft_vec with seed token (required by C++ function)
        # The draft function expects draft_vec to have exactly 1 element initially
        if len(inp) > 0:
            draft_vec.push_back(inp[-1])  # Use last token as seed
        else:
            draft_vec.push_back(0)  # Use 0 as seed if no input

        # Clamp to valid range
        ngram_min = max(ngram_cache.LLAMA_NGRAM_MIN, min(ngram_min, ngram_cache.LLAMA_NGRAM_MAX))
        ngram_max = max(ngram_cache.LLAMA_NGRAM_MIN, min(ngram_max, ngram_cache.LLAMA_NGRAM_MAX))

        ngram_cache.common_ngram_cache_draft(
            inp_vec,
            draft_vec,
            n_draft,
            ngram_min,
            ngram_max,
            ctx_cache.ptr[0],
            dyn_cache.ptr[0],
            sta_cache.ptr[0]
        )

        # Convert C++ vector to Python list
        # Skip the first element (seed token) and return only the drafted tokens
        result = []
        for i in range(1, draft_vec.size()):  # Start from 1 to skip seed
            result.append(draft_vec[i])

        return result

    def save(self, filename):
        """
        Save the n-gram cache to a file.

        Args:
            filename: Path where to save the cache

        Example:
            cache = NgramCache()
            cache.update([1, 2, 3, 4, 5])
            cache.save("my_cache.bin")
        """
        cdef std_string fname = filename.encode('utf-8')
        ngram_cache.common_ngram_cache_save(self.ptr[0], fname)

    @staticmethod
    def load(filename):
        """
        Load an n-gram cache from a file.

        Args:
            filename: Path from which to load the cache

        Returns:
            NgramCache instance with loaded data

        Example:
            cache = NgramCache.load("my_cache.bin")
            draft = cache.draft([1, 2], n_draft=5)
        """
        cdef NgramCache cache = NgramCache.__new__(NgramCache)
        cdef std_string fname = filename.encode('utf-8')

        cache.ptr = new ngram_cache.common_ngram_cache(
            ngram_cache.common_ngram_cache_load(fname)
        )
        cache.owner = True

        return cache

    def merge(self, other):
        """
        Merge another n-gram cache into this one.

        Args:
            other: Another NgramCache to merge into this cache

        Example:
            cache1 = NgramCache()
            cache1.update([1, 2, 3])

            cache2 = NgramCache()
            cache2.update([4, 5, 6])

            cache1.merge(cache2)  # cache1 now contains both patterns
        """
        if not isinstance(other, NgramCache):
            raise TypeError("Can only merge with another NgramCache")

        cdef NgramCache other_cache = other
        ngram_cache.common_ngram_cache_merge(self.ptr[0], other_cache.ptr[0])

    def __repr__(self):
        return f"<NgramCache at {hex(id(self))}>"



