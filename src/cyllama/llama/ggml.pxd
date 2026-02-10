# distutils: language=c++

from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t, uint64_t, uint8_t, uint16_t
from libc.stdio cimport FILE
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.memory cimport unique_ptr
from libcpp.set cimport set as std_set

#------------------------------------------------------------------------------
# constants

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
    GGML_ROPE_TYPE_IMROPE = 40

#------------------------------------------------------------------------------
# ggml.h

cdef extern from "ggml.h":

    ctypedef enum ggml_status:
        GGML_STATUS_ALLOC_FAILED = -2
        GGML_STATUS_FAILED = -1
        GGML_STATUS_SUCCESS = 0
        GGML_STATUS_ABORTED = 1

    ctypedef uint16_t ggml_fp16_t
    ctypedef struct ggml_bf16_t:
        uint16_t bits

    ctypedef struct ggml_context: pass
    ctypedef struct ggml_object: pass
    ctypedef struct ggml_cgraph: pass

    cdef enum ggml_type:
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
        GGML_TYPE_MXFP4 = 39 # MXFP4 (1 block)
        GGML_TYPE_COUNT = 40


    cdef enum ggml_prec:
        GGML_PREC_DEFAULT =  0
        GGML_PREC_F32     = 10

    cdef enum ggml_op:
        GGML_OP_NONE = 0

        GGML_OP_DUP
        GGML_OP_ADD
        GGML_OP_ADD_ID
        GGML_OP_ADD1
        GGML_OP_ACC
        GGML_OP_SUB
        GGML_OP_MUL
        GGML_OP_DIV
        GGML_OP_SQR
        GGML_OP_SQRT
        GGML_OP_LOG
        GGML_OP_SUM
        GGML_OP_SUM_ROWS
        GGML_OP_CUMSUM
        GGML_OP_MEAN
        GGML_OP_ARGMAX
        GGML_OP_REPEAT
        GGML_OP_REPEAT_BACK
        GGML_OP_CONCAT
        GGML_OP_SILU_BACK
        GGML_OP_NORM # normalize
        GGML_OP_RMS_NORM
        GGML_OP_RMS_NORM_BACK
        GGML_OP_GROUP_NORM

        GGML_OP_MUL_MAT
        GGML_OP_MUL_MAT_ID
        GGML_OP_OUT_PROD

        GGML_OP_SCALE
        GGML_OP_SET
        GGML_OP_CPY
        GGML_OP_CONT
        GGML_OP_RESHAPE
        GGML_OP_VIEW
        GGML_OP_PERMUTE
        GGML_OP_TRANSPOSE
        GGML_OP_GET_ROWS
        GGML_OP_GET_ROWS_BACK
        GGML_OP_SET_ROWS
        GGML_OP_DIAG
        GGML_OP_DIAG_MASK_INF
        GGML_OP_DIAG_MASK_ZERO
        GGML_OP_SOFT_MAX
        GGML_OP_SOFT_MAX_BACK
        GGML_OP_ROPE
        GGML_OP_ROPE_BACK
        GGML_OP_CLAMP
        GGML_OP_CONV_TRANSPOSE_1D
        GGML_OP_IM2COL
        GGML_OP_IM2COL_3D
        GGML_OP_CONV_3D
        GGML_OP_CONV_2D_DW
        GGML_OP_CONV_TRANSPOSE_2D
        GGML_OP_POOL_1D
        GGML_OP_POOL_2D
        GGML_OP_POOL_2D_BACK
        GGML_OP_UPSCALE
        GGML_OP_PAD
        GGML_OP_PAD_REFLECT_1D
        GGML_OP_ROLL
        GGML_OP_ARANGE
        GGML_OP_TIMESTEP_EMBEDDING
        GGML_OP_ARGSORT
        GGML_OP_TOP_K
        GGML_OP_LEAKY_RELU
        GGML_OP_TRI
        GGML_OP_FILL

        GGML_OP_FLASH_ATTN_EXT
        GGML_OP_FLASH_ATTN_BACK
        GGML_OP_SSM_CONV
        GGML_OP_SSM_SCAN
        GGML_OP_WIN_PART
        GGML_OP_WIN_UNPART
        GGML_OP_GET_REL_POS
        GGML_OP_ADD_REL_POS
        GGML_OP_RWKV_WKV6
        GGML_OP_GATED_LINEAR_ATTN
        GGML_OP_RWKV_WKV7
        GGML_OP_SOLVE_TRI

        GGML_OP_UNARY

        GGML_OP_MAP_CUSTOM1
        GGML_OP_MAP_CUSTOM2
        GGML_OP_MAP_CUSTOM3

        GGML_OP_CUSTOM

        GGML_OP_CROSS_ENTROPY_LOSS
        GGML_OP_CROSS_ENTROPY_LOSS_BACK
        GGML_OP_OPT_STEP_ADAMW
        GGML_OP_OPT_STEP_SGD

        GGML_OP_GLU

        GGML_OP_COUNT

    cdef enum ggml_unary_op:
        GGML_UNARY_OP_ABS
        GGML_UNARY_OP_SGN
        GGML_UNARY_OP_NEG
        GGML_UNARY_OP_STEP
        GGML_UNARY_OP_TANH
        GGML_UNARY_OP_ELU
        GGML_UNARY_OP_RELU
        GGML_UNARY_OP_SIGMOID
        GGML_UNARY_OP_GELU
        GGML_UNARY_OP_GELU_QUICK
        GGML_UNARY_OP_SILU
        GGML_UNARY_OP_HARDSWISH
        GGML_UNARY_OP_HARDSIGMOID
        GGML_UNARY_OP_EXP
        GGML_UNARY_OP_EXPM1
        GGML_UNARY_OP_SOFTPLUS
        GGML_UNARY_OP_GELU_ERF
        GGML_UNARY_OP_XIELU
        GGML_UNARY_OP_FLOOR
        GGML_UNARY_OP_CEIL
        GGML_UNARY_OP_ROUND
        GGML_UNARY_OP_TRUNC
        GGML_UNARY_OP_COUNT

    cdef enum ggml_tri_type:
        GGML_TRI_TYPE_UPPER_DIAG = 0
        GGML_TRI_TYPE_UPPER = 1
        GGML_TRI_TYPE_LOWER_DIAG = 2
        GGML_TRI_TYPE_LOWER = 3

    cdef enum ggml_scale_mode:
        GGML_SCALE_MODE_NEAREST = 0
        GGML_SCALE_MODE_BILINEAR = 1
        GGML_SCALE_MODE_BICUBIC = 2
        GGML_SCALE_MODE_COUNT

    cdef enum ggml_log_level:
        GGML_LOG_LEVEL_NONE  = 0
        GGML_LOG_LEVEL_INFO  = 1
        GGML_LOG_LEVEL_WARN  = 2
        GGML_LOG_LEVEL_ERROR = 3
        GGML_LOG_LEVEL_DEBUG = 4
        GGML_LOG_LEVEL_CONT  = 5

    ctypedef void (*ggml_log_callback)(ggml_log_level level, const char * text, void * user_data)
    ctypedef bint (*ggml_abort_callback)(void * data)

    cdef const char * ggml_version()
    cdef const char * ggml_commit()


    # -------------------------------------------------------------------------
    # n-dimensional tensor

    ctypedef struct ggml_tensor:
        ggml_type type

        ggml_backend_buffer * buffer

        int64_t ne[GGML_MAX_DIMS]  # number of elements
        size_t  nb[GGML_MAX_DIMS]  # stride in bytes:
                                   # nb[0] = ggml_type_size(type)
                                   # nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   # nb[i] = nb[i-1] * ne[i-1]

        # compute data
        ggml_op op

        # op params - allocated as int32_t for alignment
        int32_t op_params[16] # GGML_MAX_OP_PARAMS / sizeof(int32_t?)

        int32_t flags

        ggml_tensor * grad
        ggml_tensor * src[GGML_MAX_SRC]

        # source tensor and offset for views
        ggml_tensor * view_src
        size_t view_offs

        void * data

        char name[GGML_MAX_NAME]

        void * extra # extra things e.g. for ggml-cuda.cu

        # char padding[4]

    # -------------------------------------------------------------------------
    # ggml threadpool

    cdef enum ggml_sched_priority:
        GGML_SCHED_PRIO_NORMAL
        GGML_SCHED_PRIO_MEDIUM
        GGML_SCHED_PRIO_HIGH
        GGML_SCHED_PRIO_REALTIME

    # Threadpool params
    # Use ggml_threadpool_params_default() or ggml_threadpool_params_init() to populate the defaults
    ctypedef struct ggml_threadpool_params:
        bint                cpumask[GGML_MAX_N_THREADS] # mask of cpu cores (all-zeros means use default affinity settings)
        int                 n_threads                   # number of threads
        ggml_sched_priority prio                        # thread priority
        uint32_t            poll                        # polling level (0 - no polling, 100 - aggressive polling)
        bint                strict_cpu                  # strict cpu placement
        bint                paused                      # start in paused state

    ctypedef struct ggml_threadpool:
        pass

    ctypedef ggml_threadpool * ggml_threadpool_t

    cdef ggml_threadpool_params ggml_threadpool_params_default(int n_threads)
    cdef void ggml_threadpool_params_init(ggml_threadpool_params * p, int n_threads)
    cdef bint ggml_threadpool_params_match(const ggml_threadpool_params * p0, const ggml_threadpool_params * p1)

#------------------------------------------------------------------------------
# ggml-backend.h

cdef extern from "ggml-backend.h":

    ctypedef struct ggml_backend_buffer_type: pass
    ctypedef struct ggml_backend_buffer: pass
    ctypedef struct ggml_backend_event: pass
    ctypedef struct ggml_backend: pass
    ctypedef struct ggml_backend_reg: pass
    ctypedef struct ggml_backend_device: pass

    ctypedef ggml_backend_buffer_type * ggml_backend_buffer_type_t
    ctypedef ggml_backend_buffer * ggml_backend_buffer_t
    ctypedef ggml_backend_event * ggml_backend_event_t
    ctypedef ggml_backend * ggml_backend_t
    ctypedef void * ggml_backend_graph_plan_t
    ctypedef ggml_backend_reg * ggml_backend_reg_t
    ctypedef ggml_backend_device * ggml_backend_dev_t

    cdef enum ggml_backend_dev_type:
        # CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU
        # GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU
        # integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU
        # accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL

    # functionality supported by the device
    ctypedef struct ggml_backend_dev_caps:
        # asynchronous operations
        bint async
        # pinned host buffer
        bint host_buffer
        # creating buffers from host ptr
        bint buffer_from_host_ptr
        # event synchronization
        bint events

    # all the device properties
    ctypedef struct ggml_backend_dev_props:
        # device name
        const char * name
        # device description
        const char * description
        # device free memory in bytes
        size_t memory_free
        # device total memory in bytes
        size_t memory_total
        # device type
        ggml_backend_dev_type type
        # device id
        #   for PCI devices, this should be the PCI bus id formatted as "domain:bus:device.function" (e.g. "0000:01:00.0")
        #   if the id is unknown, this should be NULL
        const char * device_id
        # device capabilities
        ggml_backend_dev_caps caps


    cdef const char *               ggml_backend_dev_name(ggml_backend_dev_t device)
    cdef const char *               ggml_backend_dev_description(ggml_backend_dev_t device)
    cdef void                       ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total)
    cdef ggml_backend_dev_type      ggml_get_backend_dev_type "ggml_backend_dev_type" (ggml_backend_dev_t device)
    cdef void                       ggml_backend_dev_get_props(ggml_backend_dev_t device, ggml_backend_dev_props * props)
    cdef ggml_backend_reg_t         ggml_backend_dev_backend_reg(ggml_backend_dev_t device)
    cdef ggml_backend_t             ggml_backend_dev_init(ggml_backend_dev_t device, const char * params)
    cdef ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device)
    cdef ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device)
    cdef ggml_backend_buffer_t      ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size)



    ctypedef bint (*ggml_backend_sched_eval_callback)(ggml_tensor * t, bint ask, void * user_data)

    # ctypedef struct ggml_backend_device: pass

    # ctypedef ggml_backend_device * ggml_backend_dev_t

    cdef void ggml_backend_register(ggml_backend_reg_t reg)
    cdef void ggml_backend_device_register(ggml_backend_dev_t device)

    # Backend (reg) enumeration
    cdef size_t             ggml_backend_reg_count()
    cdef ggml_backend_reg_t ggml_backend_reg_get(size_t index)
    cdef ggml_backend_reg_t ggml_backend_reg_by_name(const char * name)

    # Device enumeration
    cdef size_t             ggml_backend_dev_count()
    cdef ggml_backend_dev_t ggml_backend_dev_get(size_t index)
    cdef ggml_backend_dev_t ggml_backend_dev_by_name(const char * name)
    cdef ggml_backend_dev_t ggml_backend_dev_by_type(ggml_backend_dev_type type)

    # Direct backend (stream) initialization
    cdef ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params)
    cdef ggml_backend_t ggml_backend_init_by_type(ggml_backend_dev_type type, const char * params)
    cdef ggml_backend_t ggml_backend_init_best()

    # Load a backend from a dynamic library and register it
    cdef ggml_backend_reg_t ggml_backend_load(const char * path)
    # Unload a backend if loaded dynamically and unregister it
    cdef void ggml_backend_unload(ggml_backend_reg_t reg)
    # Load all known backends from dynamic libraries
    cdef void ggml_backend_load_all()
    cdef void ggml_backend_load_all_from_path(const char * dir_path)

    ctypedef struct ggml_backend_sched: pass # FIXME: find the struct!!
    ctypedef ggml_backend_sched * ggml_backend_sched_t

#------------------------------------------------------------------------------
# ggml-cpu.h

cdef extern from "ggml-cpu.h":

    ctypedef struct ggml_cplan:
        size_t    work_size # size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data # work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads
        ggml_threadpool * threadpool

        # abort ggml_graph_compute when true
        ggml_abort_callback abort_callback
        void *              abort_callback_data

        # use only reference implementations
        bint use_ref

    cdef enum ggml_numa_strategy:
        GGML_NUMA_STRATEGY_DISABLED   = 0
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1
        GGML_NUMA_STRATEGY_ISOLATE    = 2
        GGML_NUMA_STRATEGY_NUMACTL    = 3
        GGML_NUMA_STRATEGY_MIRROR     = 4
        GGML_NUMA_STRATEGY_COUNT

    cdef void    ggml_numa_init(ggml_numa_strategy numa); # call once for better performance on NUMA systems
    cdef bint    ggml_is_numa() # true if init detected that system has >1 NUMA node

    cdef ggml_tensor * ggml_new_i32(ggml_context * ctx, int32_t value)
    cdef ggml_tensor * ggml_new_f32(ggml_context * ctx, float value)

    cdef ggml_tensor * ggml_set_i32 (ggml_tensor * tensor, int32_t value)
    cdef ggml_tensor * ggml_set_f32 (ggml_tensor * tensor, float value)

    cdef int32_t ggml_get_i32_1d(const ggml_tensor * tensor, int i)
    cdef void    ggml_set_i32_1d(const ggml_tensor * tensor, int i, int32_t value)

    cdef int32_t ggml_get_i32_nd(const ggml_tensor * tensor, int i0, int i1, int i2, int i3)
    cdef void    ggml_set_i32_nd(const ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value)

    cdef float   ggml_get_f32_1d(const ggml_tensor * tensor, int i)
    cdef void    ggml_set_f32_1d(const ggml_tensor * tensor, int i, float value)

    cdef float   ggml_get_f32_nd(const ggml_tensor * tensor, int i0, int i1, int i2, int i3)
    cdef void    ggml_set_f32_nd(const ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value)

    cdef ggml_threadpool * ggml_threadpool_new (ggml_threadpool_params  * params)
    cdef void    ggml_threadpool_free          (ggml_threadpool * threadpool)
    cdef int     ggml_threadpool_get_n_threads (ggml_threadpool * threadpool)
    cdef void    ggml_threadpool_pause         (ggml_threadpool * threadpool)
    cdef void    ggml_threadpool_resume        (ggml_threadpool * threadpool)

    # ggml_graph_plan() has to be called before ggml_graph_compute()
    # when plan.work_size > 0, caller must allocate memory for plan.work_data
    cdef ggml_cplan ggml_graph_plan(
        const ggml_cgraph * cgraph,
        int   n_threads, # = GGML_DEFAULT_N_THREADS
        ggml_threadpool * threadpool ) # = NULL
    cdef ggml_status ggml_graph_compute(ggml_cgraph * cgraph, ggml_cplan * cplan)

    # same as ggml_graph_compute() but the work data is allocated as a part of the context
    # note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
    cdef ggml_status ggml_graph_compute_with_ctx( ggml_context * ctx, ggml_cgraph * cgraph, int n_threads)

    #
    # system info
    #

    # x86
    cdef int ggml_cpu_has_sse3       ()
    cdef int ggml_cpu_has_ssse3      ()
    cdef int ggml_cpu_has_avx        ()
    cdef int ggml_cpu_has_avx_vnni   ()
    cdef int ggml_cpu_has_avx2       ()
    cdef int ggml_cpu_has_bmi2       ()
    cdef int ggml_cpu_has_f16c       ()
    cdef int ggml_cpu_has_fma        ()
    cdef int ggml_cpu_has_avx512     ()
    cdef int ggml_cpu_has_avx512_vbmi()
    cdef int ggml_cpu_has_avx512_vnni()
    cdef int ggml_cpu_has_avx512_bf16()
    cdef int ggml_cpu_has_amx_int8   ()
    # ARM
    cdef int ggml_cpu_has_neon       ()
    cdef int ggml_cpu_has_arm_fma    ()
    cdef int ggml_cpu_has_fp16_va    ()
    cdef int ggml_cpu_has_dotprod    ()
    cdef int ggml_cpu_has_matmul_int8()
    cdef int ggml_cpu_has_sve        ()
    cdef int ggml_cpu_get_sve_cnt    ()  # sve vector length in bytes
    cdef int ggml_cpu_has_sme        ()
    # other
    cdef int ggml_cpu_has_riscv_v    ()
    cdef int ggml_cpu_has_vsx        ()
    cdef int ggml_cpu_has_vxe        ()
    cdef int ggml_cpu_has_wasm_simd  ()
    cdef int ggml_cpu_has_llamafile  ()

    #
    # CPU backend
    #

    cdef ggml_backend_t ggml_backend_cpu_init()

    cdef bint ggml_backend_is_cpu                (ggml_backend_t backend)
    cdef void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads)
    cdef void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool)
    cdef void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data)

    cdef void ggml_backend_cpu_set_use_ref(ggml_backend_t backend_cpu, bint use_ref)

    cdef ggml_backend_reg_t ggml_backend_cpu_reg()

    cdef void ggml_cpu_fp32_to_fp32(const float *,       float *, int64_t)
    cdef void ggml_cpu_fp32_to_i32 (const float *,     int32_t *, int64_t)
    cdef void ggml_cpu_fp32_to_fp16(const float *, ggml_fp16_t *, int64_t)
    cdef void ggml_cpu_fp16_to_fp32(const ggml_fp16_t *, float *, int64_t)
    cdef void ggml_cpu_fp32_to_bf16(const float *, ggml_bf16_t *, int64_t)
    cdef void ggml_cpu_bf16_to_fp32(const ggml_bf16_t *, float *, int64_t)
# 
#------------------------------------------------------------------------------
# ggml-opt.h

cdef extern from "ggml-opt.h":

    ctypedef struct ggml_opt_dataset: pass
    ctypedef struct ggml_opt_context: pass
    ctypedef struct ggml_opt_result: pass

    ctypedef ggml_opt_dataset * ggml_opt_dataset_t
    ctypedef ggml_opt_context * ggml_opt_context_t
    ctypedef ggml_opt_result  * ggml_opt_result_t

    # ====== Loss ======

    # built-in loss types, i.e. the built-in quantities minimized by the optimizer
    # custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
    ctypedef enum ggml_opt_loss_type:
        GGML_OPT_LOSS_TYPE_MEAN
        GGML_OPT_LOSS_TYPE_SUM
        GGML_OPT_LOSS_TYPE_CROSS_ENTROPY
        GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR

    # ====== Dataset ======

    cdef ggml_opt_dataset_t ggml_opt_dataset_init(
            ggml_type type_data,    # the type for the internal data tensor
            ggml_type type_label,   # the type for the internal labels tensor
            int64_t   ne_datapoint, # number of elements per datapoint
            int64_t   ne_label,     # number of elements per label
            int64_t   ndata,        # total number of datapoints/labels
            int64_t   ndata_shard)  # number of datapoints/labels per shard (unit at which the dataset is shuffled/copied)
    
    cdef void ggml_opt_dataset_free(ggml_opt_dataset_t dataset)

    # get underlying tensors that store the data
    cdef int64_t              ggml_opt_dataset_ndata (ggml_opt_dataset_t dataset)
    cdef ggml_tensor * ggml_opt_dataset_data  (ggml_opt_dataset_t dataset) # shape = [ne_datapoint, ndata]
    cdef ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset) # shape = [nd_label,     ndata]

    # shuffle idata first datapoints from dataset with RNG from opt_ctx, shuffle all datapoints if idata is negative
    cdef void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata)

    # get batch at position ibatch from dataset and copy the data to data_batch and labels_batch
    cdef void ggml_opt_dataset_get_batch(
            ggml_opt_dataset_t   dataset,
            ggml_tensor * data_batch,   # shape = [ne_datapoint, ndata_batch]
            ggml_tensor * labels_batch, # shape = [ne_label,     ndata_batch]
            int64_t       ibatch)

    cdef void ggml_opt_dataset_get_batch_host(
            ggml_opt_dataset_t   dataset,
            void               * data_batch,
            size_t               nb_data_batch,
            void               * labels_batch,
            int64_t              ibatch)

    # ====== Model / Context ======

    ctypedef enum ggml_opt_build_type:
        GGML_OPT_BUILD_TYPE_FORWARD = 10
        GGML_OPT_BUILD_TYPE_GRAD    = 20
        GGML_OPT_BUILD_TYPE_OPT     = 30

    ctypedef enum ggml_opt_optimizer_type:
        GGML_OPT_OPTIMIZER_TYPE_ADAMW
        GGML_OPT_OPTIMIZER_TYPE_SGD
        GGML_OPT_OPTIMIZER_TYPE_COUNT

    ctypedef struct adamw_t:
        float alpha # learning rate
        float beta1
        float beta2
        float eps   # epsilon for numerical stability
        float wd    # weight decay for AdamW, use 0.0f to disable

    # parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
    ctypedef struct ggml_opt_optimizer_params:
        # AdamW optimizer parameters        
        adamw_t adamw

        # struct {
        #     float alpha # learning rate
        #     float beta1
        #     float beta2
        #     float eps   # epsilon for numerical stability
        #     float wd    # weight decay for AdamW, use 0.0f to disable
        # } adamw

    # callback to calculate optimizer parameters prior to a backward pass
    # userdata can be used to pass arbitrary data
    ctypedef ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata)

    # returns the default optimizer params (constant, hard-coded values)
    # userdata is not used
    cdef ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata)

    # casts userdata to ggml_opt_optimizer_params and returns it
    cdef ggml_opt_optimizer_params ggml_opt_get_constant_optimizer_params(void * userdata)

    # parameters for initializing a new optimization context
    ctypedef struct ggml_opt_params:
        ggml_backend_sched_t backend_sched # defines which backends are used to construct the compute graphs

        # by default the forward graph needs to be reconstructed for each eval
        # if ctx_compute, inputs, and outputs are set the graphs are instead allocated statically
        ggml_context * ctx_compute
        ggml_tensor  * inputs
        ggml_tensor  * outputs

        ggml_opt_loss_type  loss_type
        ggml_opt_build_type build_type

        int32_t opt_period # after how many gradient accumulation steps an optimizer step should be done

        ggml_opt_get_optimizer_params get_opt_pars # callback for calculating optimizer parameters
        void * get_opt_pars_ud                     # userdata for calculating optimizer parameters


    # get parameters for an optimization context with defaults set where possible
    # parameters for which no sensible defaults exist are supplied as arguments to this function
    cdef ggml_opt_params ggml_opt_default_params(
            ggml_backend_sched_t    backend_sched,
            ggml_opt_loss_type loss_type)

    cdef ggml_opt_context_t ggml_opt_init(ggml_opt_params params)
    cdef void ggml_opt_free(ggml_opt_context_t opt_ctx)

    # set gradients to zero, initilize loss, and optionally reset the optimizer
    cdef void ggml_opt_reset(ggml_opt_context_t opt_ctx, bint optimizer)

    # get underlying tensors that store data
    # if not using static graphs these pointers become invalid with the next call to ggml_opt_alloc
    cdef ggml_tensor * ggml_opt_inputs(  ggml_opt_context_t opt_ctx) # forward graph input tensor
    cdef ggml_tensor * ggml_opt_outputs( ggml_opt_context_t opt_ctx) # forward graph output tensor
    cdef ggml_tensor * ggml_opt_labels(  ggml_opt_context_t opt_ctx) # labels to compare outputs against
    cdef ggml_tensor * ggml_opt_loss(    ggml_opt_context_t opt_ctx) # scalar tensor that contains the loss
    cdef ggml_tensor * ggml_opt_pred(    ggml_opt_context_t opt_ctx) # predictions made by outputs
    cdef ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx) # number of matching predictions between outputs and labels

    # get the gradient accumulator for a node from the forward graph
    cdef ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, ggml_tensor * node)

    # ====== Optimization Result ======

    cdef ggml_opt_result_t ggml_opt_result_init()
    cdef void ggml_opt_result_free(ggml_opt_result_t result)
    cdef void ggml_opt_result_reset(ggml_opt_result_t result)

    # get data from result, uncertainties are optional and can be ignored by passing NULL
    cdef void ggml_opt_result_ndata(   ggml_opt_result_t result, int64_t * ndata)                  # writes 1 value, number of datapoints
    cdef void ggml_opt_result_loss(    ggml_opt_result_t result, double  * loss,     double * unc) # writes 1 value
    cdef void ggml_opt_result_pred(    ggml_opt_result_t result, int32_t * pred)                   # writes ndata values
    cdef void ggml_opt_result_accuracy(ggml_opt_result_t result, double  * accuracy, double * unc) # writes 1 value

    # ====== Computation ======

    # if not using static graphs, this function must be called prior to ggml_opt_alloc
    cdef void ggml_opt_prepare_alloc(
        ggml_opt_context_t    opt_ctx,
        ggml_context * ctx_compute,
        ggml_cgraph  * gf,
        ggml_tensor  * inputs,
        ggml_tensor  * outputs)

    # allocate the next graph for evaluation, either forward or forward + backward
    # must be called exactly once prior to calling ggml_opt_eval
    cdef void ggml_opt_alloc(ggml_opt_context_t opt_ctx, bint backward)

    # do forward pass, increment result if not NULL, do backward pass if allocated
    cdef void ggml_opt_eval(ggml_opt_context_t opt_ctx, ggml_opt_result_t result)

    # ############################################################################
    # ## The high-level functions start here. They do not depend on any private ##
    # ## functions or structs and can be copied to and adapted for user code.   ##
    # ############################################################################

    # ====== Intended Usage ======
    #
    # 1. Select the appropriate loss for your problem.
    # 2. Create a dataset and set the data for the "data" tensor. Also set the "labels" tensor if your loss needs them.
    #    Setting the shard size to 1 will be fine, it's the granularity with which data is shuffled/loaded (bigger values are faster).
    # 3. Create a GGML graph for your model with no_alloc == true. Use two separate contexts for the tensors.
    #    The first context should contain the model parameters and inputs and be allocated statically in user code.
    #    The second context should contain all other tensors and will be (re)allocated automatically.
    #    Due to this automated allocation the data of the second context is not defined when accessed in user code.
    #    Note that the second dimension of the inputs/outputs are interpreted as the number of datapoints in those tensors.
    # 4. Call ggml_opt_fit. If you need more control you can use ggml_opt_epoch instead.

    # signature for a callback while evaluating opt_ctx on dataset, called after an evaluation
    ctypedef void (*ggml_opt_epoch_callback)(
            bint               train,       # true after training evaluation, false after validation evaluation
            ggml_opt_context_t opt_ctx,
            ggml_opt_dataset_t dataset,
            ggml_opt_result_t  result,      # result associated with the dataset subsection
            int64_t            ibatch,      # number of batches that have been evaluated so far
            int64_t            ibatch_max,  # total number of batches in this dataset subsection
            int64_t            t_start_us) # time at which the evaluation on the dataset subsection was started

    # do training on front of dataset, do evaluation only on back of dataset
    cdef void ggml_opt_epoch(
            ggml_opt_context_t      opt_ctx,
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,   # result to increment during training, ignored if NULL
            ggml_opt_result_t       result_eval,    # result to increment during evaluation, ignored if NULL
            int64_t                 idata_split,    # data index at which to split training and evaluation
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval)

    # callback that prints a progress bar on stderr
    cdef void ggml_opt_epoch_callback_progress_bar(
            bint               train,
            ggml_opt_context_t opt_ctx,
            ggml_opt_dataset_t dataset,
            ggml_opt_result_t  result,
            int64_t            ibatch,
            int64_t            ibatch_max,
            int64_t            t_start_us)

    # fit model defined by inputs and outputs to dataset
    cdef void ggml_opt_fit(
            ggml_backend_sched_t            backend_sched,  # backend scheduler for constructing the compute graphs
            ggml_context                    * ctx_compute,    # context with temporarily allocated tensors to calculate the outputs
            ggml_tensor                     * inputs,         # input tensor with shape [ne_datapoint, ndata_batch]
            ggml_tensor                     * outputs,        # output tensor, must have shape [ne_label, ndata_batch] if labels are used
            ggml_opt_dataset_t              dataset,        # dataset with data and optionally also labels
            ggml_opt_loss_type              loss_type,      # loss to minimize
            ggml_opt_get_optimizer_params   get_opt_pars,   # callback to get optimizer params, userdata is pointer to epoch (of type int64_t)
            int64_t                         nepoch,         # how many times the dataset should be iterated over
            int64_t                         nbatch_logical, # datapoints optimizer step, must be a multiple of ndata_batch in inputs/outputs
            float                           val_split,      # fraction of the dataset to use for validation, must be in [0.0f, 1.0f)
            bint                            silent)         # whether or not info prints to stderr should be suppressed


    cdef int ggml_time_us()

