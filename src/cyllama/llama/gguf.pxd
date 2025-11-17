# distutils: language=c++

from libc.stdint cimport uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t
from libc.stddef cimport size_t

cimport ggml

#------------------------------------------------------------------------------
# gguf.h - GGUF file format API
#
# GGUF is the binary file format used by ggml for storing models.
# Format structure:
# 1. File magic "GGUF" (4 bytes)
# 2. File version (uint32_t)
# 3. Number of tensors (int64_t)
# 4. Number of key-value pairs (int64_t)
# 5. KV pairs (key, type, value)
# 6. Tensor metadata (name, dims, type, offset)
# 7. Tensor data (aligned)

cdef extern from "gguf.h":

    # Constants
    cdef const char * GGUF_MAGIC
    cdef int GGUF_VERSION
    cdef const char * GGUF_KEY_GENERAL_ALIGNMENT
    cdef int GGUF_DEFAULT_ALIGNMENT

    # GGUF value types
    ctypedef enum gguf_type:
        GGUF_TYPE_UINT8   = 0
        GGUF_TYPE_INT8    = 1
        GGUF_TYPE_UINT16  = 2
        GGUF_TYPE_INT16   = 3
        GGUF_TYPE_UINT32  = 4
        GGUF_TYPE_INT32   = 5
        GGUF_TYPE_FLOAT32 = 6
        GGUF_TYPE_BOOL    = 7
        GGUF_TYPE_STRING  = 8
        GGUF_TYPE_ARRAY   = 9
        GGUF_TYPE_UINT64  = 10
        GGUF_TYPE_INT64   = 11
        GGUF_TYPE_FLOAT64 = 12
        GGUF_TYPE_COUNT

    # Opaque context structure
    ctypedef struct gguf_context:
        pass

    # Initialization parameters
    ctypedef struct gguf_init_params:
        bint no_alloc
        ggml.ggml_context ** ctx

    # Context creation and destruction
    cdef gguf_context * gguf_init_empty()
    cdef gguf_context * gguf_init_from_file(const char * fname, gguf_init_params params)
    cdef void gguf_free(gguf_context * ctx)

    # Utility functions
    cdef const char * gguf_type_name(gguf_type type)

    # Context queries
    cdef uint32_t gguf_get_version(const gguf_context * ctx)
    cdef size_t gguf_get_alignment(const gguf_context * ctx)
    cdef size_t gguf_get_data_offset(const gguf_context * ctx)

    # Key-value pair queries
    cdef int64_t gguf_get_n_kv(const gguf_context * ctx)
    cdef int64_t gguf_find_key(const gguf_context * ctx, const char * key)
    cdef const char * gguf_get_key(const gguf_context * ctx, int64_t key_id)

    # KV type queries
    cdef gguf_type gguf_get_kv_type(const gguf_context * ctx, int64_t key_id)
    cdef gguf_type gguf_get_arr_type(const gguf_context * ctx, int64_t key_id)

    # Get scalar values
    cdef uint8_t gguf_get_val_u8(const gguf_context * ctx, int64_t key_id)
    cdef int8_t gguf_get_val_i8(const gguf_context * ctx, int64_t key_id)
    cdef uint16_t gguf_get_val_u16(const gguf_context * ctx, int64_t key_id)
    cdef int16_t gguf_get_val_i16(const gguf_context * ctx, int64_t key_id)
    cdef uint32_t gguf_get_val_u32(const gguf_context * ctx, int64_t key_id)
    cdef int32_t gguf_get_val_i32(const gguf_context * ctx, int64_t key_id)
    cdef float gguf_get_val_f32(const gguf_context * ctx, int64_t key_id)
    cdef uint64_t gguf_get_val_u64(const gguf_context * ctx, int64_t key_id)
    cdef int64_t gguf_get_val_i64(const gguf_context * ctx, int64_t key_id)
    cdef double gguf_get_val_f64(const gguf_context * ctx, int64_t key_id)
    cdef bint gguf_get_val_bool(const gguf_context * ctx, int64_t key_id)
    cdef const char * gguf_get_val_str(const gguf_context * ctx, int64_t key_id)
    cdef const void * gguf_get_val_data(const gguf_context * ctx, int64_t key_id)

    # Get array values
    cdef size_t gguf_get_arr_n(const gguf_context * ctx, int64_t key_id)
    cdef const void * gguf_get_arr_data(const gguf_context * ctx, int64_t key_id)
    cdef const char * gguf_get_arr_str(const gguf_context * ctx, int64_t key_id, size_t i)

    # Tensor queries
    cdef int64_t gguf_get_n_tensors(const gguf_context * ctx)
    cdef int64_t gguf_find_tensor(const gguf_context * ctx, const char * name)
    cdef size_t gguf_get_tensor_offset(const gguf_context * ctx, int64_t tensor_id)
    cdef const char * gguf_get_tensor_name(const gguf_context * ctx, int64_t tensor_id)
    cdef ggml.ggml_type gguf_get_tensor_type(const gguf_context * ctx, int64_t tensor_id)
    cdef size_t gguf_get_tensor_size(const gguf_context * ctx, int64_t tensor_id)

    # KV pair modification
    cdef int64_t gguf_remove_key(gguf_context * ctx, const char * key)

    # Set scalar values
    cdef void gguf_set_val_u8(gguf_context * ctx, const char * key, uint8_t val)
    cdef void gguf_set_val_i8(gguf_context * ctx, const char * key, int8_t val)
    cdef void gguf_set_val_u16(gguf_context * ctx, const char * key, uint16_t val)
    cdef void gguf_set_val_i16(gguf_context * ctx, const char * key, int16_t val)
    cdef void gguf_set_val_u32(gguf_context * ctx, const char * key, uint32_t val)
    cdef void gguf_set_val_i32(gguf_context * ctx, const char * key, int32_t val)
    cdef void gguf_set_val_f32(gguf_context * ctx, const char * key, float val)
    cdef void gguf_set_val_u64(gguf_context * ctx, const char * key, uint64_t val)
    cdef void gguf_set_val_i64(gguf_context * ctx, const char * key, int64_t val)
    cdef void gguf_set_val_f64(gguf_context * ctx, const char * key, double val)
    cdef void gguf_set_val_bool(gguf_context * ctx, const char * key, bint val)
    cdef void gguf_set_val_str(gguf_context * ctx, const char * key, const char * val)

    # Set array values
    cdef void gguf_set_arr_data(gguf_context * ctx, const char * key, gguf_type type, const void * data, size_t n)
    cdef void gguf_set_arr_str(gguf_context * ctx, const char * key, const char ** data, size_t n)

    # Copy KV pairs from another context
    cdef void gguf_set_kv(gguf_context * ctx, const gguf_context * src)

    # Tensor modification
    cdef void gguf_add_tensor(gguf_context * ctx, const ggml.ggml_tensor * tensor)
    cdef void gguf_set_tensor_type(gguf_context * ctx, const char * name, ggml.ggml_type type)
    cdef void gguf_set_tensor_data(gguf_context * ctx, const char * name, const void * data)

    # File writing
    cdef bint gguf_write_to_file(const gguf_context * ctx, const char * fname, bint only_meta)
    cdef size_t gguf_get_meta_size(const gguf_context * ctx)
    cdef void gguf_get_meta_data(const gguf_context * ctx, void * data)
