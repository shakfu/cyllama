# distutils: language=c++

from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t, uint64_t, uint8_t
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

#------------------------------------------------------------------------------
# ggml-cpu.h

cdef extern from "ggml-cpu.h":

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

    cdef ggml_threadpool * ggml_threadpool_new(ggml_threadpool_params * params)
    cdef void ggml_threadpool_free(ggml_threadpool * threadpool)
    cdef int ggml_threadpool_get_n_threads(ggml_threadpool * threadpool)
    cdef void ggml_threadpool_pause(ggml_threadpool * threadpool)
    cdef void ggml_threadpool_resume(ggml_threadpool * threadpool)

    cdef enum ggml_numa_strategy:
        GGML_NUMA_STRATEGY_DISABLED   = 0
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1
        GGML_NUMA_STRATEGY_ISOLATE    = 2
        GGML_NUMA_STRATEGY_NUMACTL    = 3
        GGML_NUMA_STRATEGY_MIRROR     = 4
        GGML_NUMA_STRATEGY_COUNT

#------------------------------------------------------------------------------
# ggml.h

cdef extern from "ggml.h":

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
        GGML_TYPE_COUNT = 39


    cdef enum ggml_op:
        GGML_OP_NONE = 0

        GGML_OP_DUP
        GGML_OP_ADD
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
        GGML_OP_CONV_TRANSPOSE_2D
        GGML_OP_POOL_1D
        GGML_OP_POOL_2D
        GGML_OP_UPSCALE # nearest interpolate
        GGML_OP_PAD
        GGML_OP_ARANGE
        GGML_OP_TIMESTEP_EMBEDDING
        GGML_OP_ARGSORT
        GGML_OP_LEAKY_RELU

        GGML_OP_FLASH_ATTN_EXT
        GGML_OP_FLASH_ATTN_BACK
        GGML_OP_SSM_CONV
        GGML_OP_SSM_SCAN
        GGML_OP_WIN_PART
        GGML_OP_WIN_UNPART
        GGML_OP_GET_REL_POS
        GGML_OP_ADD_REL_POS

        GGML_OP_UNARY

        GGML_OP_MAP_UNARY
        GGML_OP_MAP_BINARY

        GGML_OP_MAP_CUSTOM1_F32
        GGML_OP_MAP_CUSTOM2_F32
        GGML_OP_MAP_CUSTOM3_F32

        GGML_OP_MAP_CUSTOM1
        GGML_OP_MAP_CUSTOM2
        GGML_OP_MAP_CUSTOM3

        GGML_OP_CROSS_ENTROPY_LOSS
        GGML_OP_CROSS_ENTROPY_LOSS_BACK

        GGML_OP_COUNT


    cdef enum ggml_log_level:
        GGML_LOG_LEVEL_NONE  = 0
        GGML_LOG_LEVEL_INFO  = 1
        GGML_LOG_LEVEL_WARN  = 2
        GGML_LOG_LEVEL_ERROR = 3
        GGML_LOG_LEVEL_DEBUG = 4
        GGML_LOG_LEVEL_CONT  = 5

    ctypedef void (*ggml_log_callback)(ggml_log_level level, const char * text, void * user_data)
    ctypedef bint (*ggml_abort_callback)(void * data)


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


    cdef void    ggml_time_init() # call this once at the beginning of the program
    cdef int64_t ggml_time_ms()
    cdef int64_t ggml_time_us()
    cdef int64_t ggml_cycles()
    cdef int64_t ggml_cycles_per_ms()

    # cdef void ggml_abort(const char * file, int line, const char * fmt, ...)


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
        # accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL
    

    ctypedef bint (*ggml_backend_sched_eval_callback)(ggml_tensor * t, bint ask, void * user_data)

    # ctypedef struct ggml_backend_device: pass

    # ctypedef ggml_backend_device * ggml_backend_dev_t

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
    # = ggml_backend_dev_init(ggml_backend_dev_by_name(name), params)
    cdef ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params)
    # = ggml_backend_dev_init(ggml_backend_dev_by_type(type), params)
    cdef ggml_backend_t ggml_backend_init_by_type(ggml_backend_dev_type type, const char * params)
    # = ggml_backend_dev_init(ggml_backend_dev_by_type(GPU) OR ggml_backend_dev_by_type(CPU), NULL)
    cdef ggml_backend_t ggml_backend_init_best()

    # Load a backend from a dynamic library and register it
    cdef ggml_backend_reg_t ggml_backend_load(const char * path)
    # Unload a backend if loaded dynamically and unregister it
    cdef void ggml_backend_unload(ggml_backend_reg_t reg)
    # Load all known backends from dynamic libraries
    cdef void ggml_backend_load_all()
    cdef void ggml_backend_load_all_from_path(const char * dir_path)

    ctypedef struct ggml_backend_sched: pass # FIXME: find the struct!!
    ctypedef ggml_backend_sched * ggml_backend_sched_t;

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



#------------------------------------------------------------------------------
# llama.h

cdef extern from "llama.h":

    long LLAMA_DEFAULT_SEED
    int LLAMA_TOKEN_NULL

    ctypedef struct llama_vocab: pass
    ctypedef struct llama_model: pass
    ctypedef struct llama_context: pass
    ctypedef struct llama_sampler: pass
    ctypedef struct llama_kv_cache: pass

    ctypedef int32_t llama_pos
    ctypedef int32_t llama_token
    ctypedef int32_t llama_seq_id

    cdef enum llama_vocab_type:
        LLAMA_VOCAB_TYPE_NONE = 0  # For models without vocab
        LLAMA_VOCAB_TYPE_SPM  = 1  # LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE  = 2  # GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM  = 3  # BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM  = 4  # T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV = 5  # RWKV tokenizer based on greedy tokenization

    # pre-tokenization types
    cdef enum llama_vocab_pre_type:
        LLAMA_VOCAB_PRE_TYPE_DEFAULT
        LLAMA_VOCAB_PRE_TYPE_LLAMA3
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER
        LLAMA_VOCAB_PRE_TYPE_FALCON
        LLAMA_VOCAB_PRE_TYPE_MPT
        LLAMA_VOCAB_PRE_TYPE_STARCODER
        LLAMA_VOCAB_PRE_TYPE_GPT2
        LLAMA_VOCAB_PRE_TYPE_REFACT
        LLAMA_VOCAB_PRE_TYPE_COMMAND_R
        LLAMA_VOCAB_PRE_TYPE_STABLELM2
        LLAMA_VOCAB_PRE_TYPE_QWEN2
        LLAMA_VOCAB_PRE_TYPE_OLMO
        LLAMA_VOCAB_PRE_TYPE_DBRX
        LLAMA_VOCAB_PRE_TYPE_SMAUG
        LLAMA_VOCAB_PRE_TYPE_PORO
        LLAMA_VOCAB_PRE_TYPE_CHATGLM3
        LLAMA_VOCAB_PRE_TYPE_CHATGLM4
        LLAMA_VOCAB_PRE_TYPE_VIKING
        LLAMA_VOCAB_PRE_TYPE_JAIS
        LLAMA_VOCAB_PRE_TYPE_TEKKEN
        LLAMA_VOCAB_PRE_TYPE_SMOLLM
        LLAMA_VOCAB_PRE_TYPE_CODESHELL
        LLAMA_VOCAB_PRE_TYPE_BLOOM
        LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH
        LLAMA_VOCAB_PRE_TYPE_EXAONE
        LLAMA_VOCAB_PRE_TYPE_CHAMELEON
        LLAMA_VOCAB_PRE_TYPE_MINERVA
        LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM
        LLAMA_VOCAB_PRE_TYPE_GPT4O
        LLAMA_VOCAB_PRE_TYPE_SUPERBPE
        LLAMA_VOCAB_PRE_TYPE_TRILLION
        LLAMA_VOCAB_PRE_TYPE_BAILINGMOE
        LLAMA_VOCAB_PRE_TYPE_LLAMA4
        LLAMA_VOCAB_PRE_TYPE_MINERVA
        LLAMA_VOCAB_PRE_TYPE_SEED_CODER


    cdef enum llama_rope_type:
        LLAMA_ROPE_TYPE_NONE   = -1
        LLAMA_ROPE_TYPE_NORM   = 0
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION

    cdef enum llama_token_type:
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0
        LLAMA_TOKEN_TYPE_NORMAL       = 1
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2
        LLAMA_TOKEN_TYPE_CONTROL      = 3
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4
        LLAMA_TOKEN_TYPE_UNUSED       = 5
        LLAMA_TOKEN_TYPE_BYTE         = 6

    cdef enum llama_token_attr:
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9

    cdef enum llama_ftype:
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

    cdef enum llama_rope_scaling_type:
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE

    cdef enum llama_pooling_type:
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1
        LLAMA_POOLING_TYPE_NONE = 0
        LLAMA_POOLING_TYPE_MEAN = 1
        LLAMA_POOLING_TYPE_CLS  = 2
        LLAMA_POOLING_TYPE_LAST = 3
        LLAMA_POOLING_TYPE_RANK = 4 # used by reranking models to attach the classification head to the graph

    cdef enum llama_attention_type:
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1

    cdef enum llama_split_mode:
        LLAMA_SPLIT_MODE_NONE  = 0 # single GPU
        LLAMA_SPLIT_MODE_LAYER = 1 # split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW   = 2 # split layers and KV across GPUs, use tensor parallelism if supported

    ctypedef struct llama_token_data:
        llama_token id
        float logit
        float p

    ctypedef struct llama_token_data_array:
        # NOTE: this pointer can be modified by the samplers
        llama_token_data * data
        size_t size
        int64_t selected  # this is the index in the data array (i.e. not the token id)
        bint sorted

    ctypedef bint (*llama_progress_callback)(float progress, void * user_data)

    ctypedef struct llama_batch:
        int32_t n_tokens

        llama_token  *  token
        float        *  embd
        llama_pos    *  pos
        int32_t      *  n_seq_id
        llama_seq_id ** seq_id
        int8_t       *  logits   # TODO: rename this to "output"

    cdef enum llama_model_kv_override_type:
        LLAMA_KV_OVERRIDE_TYPE_INT
        LLAMA_KV_OVERRIDE_TYPE_FLOAT
        LLAMA_KV_OVERRIDE_TYPE_BOOL
        LLAMA_KV_OVERRIDE_TYPE_STR

    ctypedef struct llama_model_kv_override: # FLATTENED nested union enum
        llama_model_kv_override_type tag
        char key[128]
        int64_t val_i64
        double  val_f64
        bint    val_bool
        char    val_str[128]

    ctypedef struct llama_model_tensor_buft_override:
       const char * pattern
       # ggml_backend_buffer_type_t buft # TODO

    ctypedef struct llama_model_params:
        ggml_backend_dev_t * devices   # NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        # NULL-terminated list of buffer types to use for tensors that match a pattern
        const llama_model_tensor_buft_override * tensor_buft_overrides;
        int32_t n_gpu_layers           # number of layers to store in VRAM
        llama_split_mode split_mode    # how to split the model across multiple GPUs
        int32_t main_gpu               # the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        const float * tensor_split     # proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        # Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        # If the provided progress_callback returns true, model loading continues.
        # If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback
        void * progress_callback_user_data
        const llama_model_kv_override * kv_overrides
        bint vocab_only
        bint use_mmap
        bint use_mlock
        bint check_tensors

    ctypedef struct llama_context_params:
        uint32_t n_ctx             # text context, 0 = from model
        uint32_t n_batch           # logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch          # physical maximum batch size
        uint32_t n_seq_max         # max number of sequences (i.e. distinct states for recurrent models)
        uint32_t n_threads         # number of threads to use for generation
        uint32_t n_threads_batch   # number of threads to use for batch processing

        llama_rope_scaling_type rope_scaling_type # RoPE scaling type
        llama_pooling_type      pooling_type      # whether to pool (sum) embedding results by sequence id
        llama_attention_type    attention_type    # attention type to use for embeddings

        float    rope_freq_base   # RoPE base frequency, 0 = from model
        float    rope_freq_scale  # RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor  # YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor # YaRN magnitude scaling factor
        float    yarn_beta_fast   # YaRN low correction dim
        float    yarn_beta_slow   # YaRN high correction dim
        uint32_t yarn_orig_ctx    # YaRN original context size
        float    defrag_thold     # defragment the KV cache if holes/size > thold, < 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml_type type_k # data type for K cache [EXPERIMENTAL]
        ggml_type type_v # data type for V cache [EXPERIMENTAL]

        # Abort callback
        # if it returns true, execution of llama_decode() will be aborted
        # currently works only with CPU execution
        ggml_abort_callback abort_callback
        void * abort_callback_data

        # Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bint embeddings  # if true, extract embeddings (together with logits)
        bint offload_kqv # whether to offload the KQV ops (including the KV cache) to GPU
        bint flash_attn  # whether to use flash attention [EXPERIMENTAL]
        bint no_perf     # whether to measure performance timings
        bint op_offload  # whether to offload host tensor operations to device


    ctypedef struct llama_model_quantize_params:
        int32_t nthread                     # number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        llama_ftype ftype                   # quantize to this llama_ftype
        ggml_type output_tensor_type        # output tensor type
        ggml_type token_embedding_type      # itoken embeddings tensor type
        bint allow_requantize               # allow quantizing non-f32/f16 tensors
        bint quantize_output_tensor         # quantize output.weight
        bint only_copy                      # only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bint pure                           # quantize all tensors to the default type
        bint keep_split                     # quantize to the same number of shards
        void * imatrix                      # pointer to importance matrix data
        void * kv_overrides                 # pointer to vector containing overrides
        void * tensor_types                 # pointer to vector containing tensor types

    ctypedef struct llama_logit_bias:
        llama_token token
        float bias

    ctypedef struct llama_sampler_chain_params:
        bint no_perf # whether to measure performance timings

    ctypedef struct llama_chat_message:
        const char * role
        const char * content

    ctypedef struct llama_adapter_lora: pass


    # -------------------------------------------------------------------------
    # llama.h functions

    # TODO: update API to start accepting pointers to params structs (https://github.com/ggerganov/llama.cpp/discussions/9172)
    cdef llama_model_params llama_model_default_params()
    cdef llama_context_params llama_context_default_params()
    cdef llama_sampler_chain_params  llama_sampler_chain_default_params()
    cdef llama_model_quantize_params llama_model_quantize_default_params()


    # Initialize the llama + ggml backend
    # If numa is true, use NUMA optimizations
    # Call once at the start of the program
    cdef void llama_backend_init()

    # Call once at the end of the program - currently only used for MPI
    cdef void llama_backend_free()

    #optional:
    cdef void llama_numa_init(ggml_numa_strategy numa)

    # Optional: an auto threadpool gets created in ggml if not passed explicitly
    cdef void llama_attach_threadpool(llama_context * ctx, ggml_threadpool_t threadpool, ggml_threadpool_t threadpool_batch)
    cdef void llama_detach_threadpool(llama_context * ctx)

    # Load the model from a file
    # If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    # If the split file name does not follow this pattern, use llama_model_load_from_splits
    cdef llama_model * llama_model_load_from_file(
            const char * path_model,
            llama_model_params params)

    # Load the model from multiple splits (support custom naming scheme)
    # The paths must be in the correct order
    cdef llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
                     llama_model_params    params)

    cdef void llama_model_save_to_file(
            const llama_model * model,
                const char * path_model);

    cdef void llama_model_free(llama_model * model)

    cdef llama_context * llama_init_from_model(
                     llama_model * model,
            llama_context_params   params)

    # Frees all allocated memory
    cdef void llama_free(llama_context * ctx)

    cdef int64_t llama_time_us()

    cdef size_t llama_max_devices()

    cdef bint llama_supports_mmap       ()
    cdef bint llama_supports_mlock      ()
    cdef bint llama_supports_gpu_offload()
    cdef bint llama_supports_rpc()

    cdef uint32_t llama_n_ctx      (const llama_context * ctx)
    cdef uint32_t llama_n_batch    (const llama_context * ctx)
    cdef uint32_t llama_n_ubatch   (const llama_context * ctx)
    cdef uint32_t llama_n_seq_max  (const llama_context * ctx)

    cdef const llama_model * llama_get_model(const llama_context * ctx)
    cdef llama_kv_cache * llama_get_kv_self(      llama_context * ctx)
    # cdef llama_pooling_type llama_pooling_type(const llama_context * ctx) # TODO: rename to llama_get_pooling_type

    cdef llama_pooling_type  llama_get_pooling_type "llama_pooling_type" (const llama_context * ctx)
    cdef const llama_vocab * llama_model_get_vocab(const llama_model * model)
    cdef llama_rope_type     llama_get_model_rope_type "llama_model_rope_type" (const llama_model * model)

    cdef int32_t llama_model_n_ctx_train(const llama_model * model)
    cdef int32_t llama_model_n_embd     (const llama_model * model)
    cdef int32_t llama_model_n_layer    (const llama_model * model)
    cdef int32_t llama_model_n_head     (const llama_model * model)
    cdef int32_t llama_model_n_head_kv  (const llama_model * model)

    # Get the model's RoPE frequency scaling factor
    cdef float llama_model_rope_freq_scale_train(const llama_model * model)

    cdef llama_vocab_type llama_get_vocab_type "llama_vocab_type" (const llama_vocab * vocab)

    cdef int32_t llama_vocab_n_tokens(const llama_vocab * vocab)

    # Functions to access the model's GGUF metadata scalar values
    # - The functions return the length of the string on success, or -1 on failure
    # - The output string is always null-terminated and cleared on failure
    # - When retrieving a string, an extra byte must be allocated to account for the null terminator
    # - GGUF array values are not supported by these functions

    # Get metadata value as a string by key name
    cdef int32_t llama_model_meta_val_str(const llama_model * model, const char * key, char * buf, size_t buf_size)

    # Get the number of metadata key/value pairs
    cdef int32_t llama_model_meta_count(const llama_model * model)

    # Get metadata key name by index
    cdef int32_t llama_model_meta_key_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size)

    # Get metadata value as a string by index
    cdef int32_t llama_model_meta_val_str_by_index(const llama_model * model, int32_t i, char * buf, size_t buf_size)

    # Get a string describing the model type
    cdef int32_t llama_model_desc(const llama_model * model, char * buf, size_t buf_size)

    # Returns the total size of all the tensors in the model in bytes
    cdef uint64_t llama_model_size(const llama_model * model)

    # Get the default chat template. Returns nullptr if not available
    # If name is NULL, returns the default chat template
    cdef const char * llama_model_chat_template(const llama_model * model, const char * name)

    # Returns the total number of parameters in the model
    cdef uint64_t llama_model_n_params(const llama_model * model)

    # Get a llama model tensor
    # cdef ggml_tensor * llama_get_model_tensor(llama_model * model, const char * name)

    # Returns true if the model contains an encoder that requires llama_encode() call
    cdef bint llama_model_has_encoder(const llama_model * model)

    # Returns true if the model contains a decoder that requires llama_decode() call
    cdef bint llama_model_has_decoder(const llama_model * model)

    # For encoder-decoder models, this function returns id of the token that must be provided
    # to the decoder to start generating output sequence. For other models, it returns -1.
    cdef llama_token llama_model_decoder_start_token(const llama_model * model)

    # Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    cdef bint llama_model_is_recurrent(const llama_model * model)

    # Returns 0 on success
    cdef uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params)

    #
    # Adapters
    #

    # Load a LoRA adapter from file
    # TODO: rename to llama_adapter_lora_init
    cdef llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora)

    # Manually free a LoRA adapter
    # Note: loaded adapters will be free when the associated model is deleted
    cdef void llama_adapter_lora_free(llama_adapter_lora * adapter)

    # The following functions operate on a llama_context, hence the naming: llama_verb_...

    # Add a loaded LoRA adapter to given context
    # This will not modify model's weight
    cdef int32_t llama_set_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter,
            float scale)

    # Remove a specific LoRA adapter from given context
    # Return -1 if the adapter is not present in the context
    cdef int32_t llama_rm_adapter_lora(
            llama_context * ctx,
            llama_adapter_lora * adapter)

    # Remove all LoRA adapters from given context
    cdef void llama_clear_adapter_lora(llama_context * ctx)

    # Apply a loaded control vector to a llama_context, or if data is NULL, clear
    # the currently loaded vector.
    # n_embd should be the size of a single layer's control, and data should point
    # to an n_embd x n_layers buffer starting from layer 1.
    # il_start and il_end are the layer range the vector should apply to (both inclusive)
    # See llama_control_vector_load in common to load a control vector.
    cdef int32_t llama_apply_adapter_cvec(
            llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end)


    # -------------------------------------------------------------------------
    # KV cache

    # TODO: start using struct llama_kv_cache

    ctypedef struct llama_kv_cache_view_cell:
        # The position for this cell. Takes KV cache shifts into account.
        # May be negative if the cell is not populated.
        llama_pos pos

    # An updateable view of the KV cache.
    ctypedef struct llama_kv_cache_view:
        # Number of KV cache cells. This will be the same as the context size.
        int32_t n_cells

        # Maximum number of sequences that can exist in a cell. It's not an error
        # if there are more sequences in a cell than this value, however they will
        # not be visible in the view cells_sequences.
        int32_t n_seq_max

        # Number of tokens in the cache. For example, if there are two populated
        # cells, the first with 1 sequence id in it and the second with 2 sequence
        # ids then you'll have 3 tokens.
        int32_t token_count

        # Number of populated cache cells.
        int32_t used_cells

        # Maximum contiguous empty slots in the cache.
        int32_t max_contiguous

        # Index to the start of the max_contiguous slot range. Can be negative
        # when cache is full.
        int32_t max_contiguous_idx

        # Information for an individual cell.
        llama_kv_cache_view_cell * cells

        # The sequences for each cell. There will be n_seq_max items per cell.
        llama_seq_id * cells_sequences

    # Create an empty KV cache view. (use only for debugging purposes)
    cdef llama_kv_cache_view llama_kv_cache_view_init(const llama_context * ctx, int32_t n_seq_max)

    # Free a KV cache view. (use only for debugging purposes)
    cdef void llama_kv_cache_view_free(llama_kv_cache_view * view)

    # Update the KV cache view structure with the current state of the KV cache. (use only for debugging purposes)
    cdef void llama_kv_cache_view_update(const llama_context * ctx, llama_kv_cache_view * view)

    # Returns the number of tokens in the KV cache (slow, use only for debug)
    # If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    cdef int32_t llama_kv_self_n_tokens(const llama_context * ctx)

    # Returns the number of used KV cells (i.e. have at least one sequence assigned to them)
    cdef int32_t llama_kv_self_used_cells(const llama_context * ctx)

    # Clear the KV cache - both cell info is erased and KV data is zeroed
    cdef void llama_kv_self_clear(llama_context * ctx)


    # Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    # Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    # seq_id < 0 : match any sequence
    # p0 < 0     : [0,  p1]
    # p1 < 0     : [p0, inf)
    cdef bint llama_kv_self_seq_rm(
            llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1)

    # Copy all tokens that belong to the specified sequence to another sequence
    # Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
    # p0 < 0 : [0,  p1]
    # p1 < 0 : [p0, inf)
    cdef void llama_kv_self_seq_cp(
            llama_context * ctx,
                    llama_seq_id   seq_id_src,
                    llama_seq_id   seq_id_dst,
                       llama_pos   p0,
                       llama_pos   p1)

    # Removes all tokens that do not belong to the specified sequence
    cdef void llama_kv_self_seq_keep(
            llama_context * ctx,
                    llama_seq_id   seq_id)

    # Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    # If the KV cache is RoPEd, the KV data is updated accordingly:
    #   - lazily on next llama_decode()
    #   - explicitly with llama_kv_self_update()
    # p0 < 0 : [0,  p1]
    # p1 < 0 : [p0, inf)
    cdef void llama_kv_self_seq_add(
            llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                       llama_pos   delta)

    # Integer division of the positions by factor of `d > 1`
    # If the KV cache is RoPEd, the KV data is updated accordingly:
    #   - lazily on next llama_decode()
    #   - explicitly with llama_kv_self_update()
    # p0 < 0 : [0,  p1]
    # p1 < 0 : [p0, inf)
    cdef void llama_kv_self_seq_div(
            llama_context * ctx,
                    llama_seq_id   seq_id,
                       llama_pos   p0,
                       llama_pos   p1,
                             int   d)

    # Returns the largest position present in the KV cache for the specified sequence
    cdef llama_pos llama_kv_self_seq_pos_max(
            llama_context * ctx,
                    llama_seq_id   seq_id)

    # Defragment the KV cache
    # This will be applied:
    #   - lazily on next llama_decode()
    #   - explicitly with llama_kv_self_update()
    cdef void llama_kv_self_defrag(llama_context * ctx)

    # Apply the KV cache updates (such as K-shifts, defragmentation, etc.)
    cdef void llama_kv_self_update(llama_context * ctx)

    # Check if the context supports KV cache shifting
    cdef bint llama_kv_self_can_shift(llama_context * ctx)

    # -------------------------------------------------------------------------
    # State / sessions

    # Returns the *actual* size in bytes of the state
    # (logits, embedding and kv_cache)
    # Only use when saving the state, not when restoring it, otherwise the size may be too small.
    cdef size_t llama_state_get_size( llama_context * ctx)

    # Copies the state to the specified destination address.
    # Destination needs to have allocated enough memory.
    # Returns the number of bytes copied
    cdef size_t llama_state_get_data(
             llama_context * ctx,
                         uint8_t * dst,
                          size_t   size)

    # Set the state reading from the specified address
    # Returns the number of bytes read
    cdef size_t llama_state_set_data(
             llama_context * ctx,
                   const uint8_t * src,
                          size_t   size)

    # Save/load session file
    cdef bint llama_state_load_file(
             llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out)

    cdef bint llama_state_save_file(
             llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count)

    # Get the exact size needed to copy the KV cache of a single sequence
    cdef size_t llama_state_seq_get_size(
             llama_context * ctx,
                    llama_seq_id   seq_id)

    # Copy the KV cache of a single sequence into the specified buffer
    cdef size_t llama_state_seq_get_data(
             llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id)

    # Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    # Returns:
    #  - Positive: Ok
    #  - Zero: Failed to load
    cdef size_t llama_state_seq_set_data(
             llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id)

    cdef size_t llama_state_seq_save_file(
             llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count)

    cdef size_t llama_state_seq_load_file(
             llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out)

    # -------------------------------------------------------------------------
    # Decoding

    # Return batch for single sequence of tokens
    # The sequence ID will be fixed to 0
    # The position of the tokens will be tracked automatically by llama_decode
    #
    # NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    #
    cdef llama_batch llama_batch_get_one(llama_token * tokens, int32_t n_tokens)


    # Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    # Each token can be assigned up to n_seq_max sequence ids
    # The batch has to be freed with llama_batch_free()
    # If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    # Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    # The rest of the llama_batch members are allocated with size n_tokens
    # All members are left uninitialized
    cdef llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max)

    # Frees a batch of tokens allocated with llama_batch_init()
    cdef void llama_batch_free( llama_batch batch)

    # Processes a batch of tokens with the ecoder part of the encoder-decoder model.
    # Stores the encoder output internally for later use by the decoder cross-attention layers.
    #   0 - success
    # < 0 - error. the KV cache state is restored to the state before this call
    cdef int32_t llama_encode(llama_context * ctx, llama_batch batch)

    # Positive return values does not mean a fatal error, but rather a warning.
    #   0 - success
    #   1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    # < 0 - error. the KV cache state is restored to the state before this call
    cdef int32_t llama_decode(llama_context * ctx, llama_batch batch)

    # Set the number of threads used for decoding
    # n_threads is the number of threads used for generation (single token)
    # n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    cdef void llama_set_n_threads( llama_context * ctx, uint32_t n_threads, uint32_t n_threads_batch)

    # Get the number of threads used for generation of a single token.
    cdef uint32_t llama_n_threads( llama_context * ctx)

    # Get the number of threads used for prompt and batch processing (multiple token).
    cdef uint32_t llama_n_threads_batch( llama_context * ctx)

    # Set whether the model is in embeddings mode or not
    # If true, embeddings will be returned but logits will not
    cdef void llama_set_embeddings( llama_context * ctx, bint embeddings)

    # Set whether to use causal attention or not
    # If set to true, the model will only attend to the past tokens
    cdef void llama_set_causal_attn( llama_context * ctx, bint causal_attn)

    # Set abort callback
    cdef void llama_set_abort_callback( llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data)

    # Wait until all computations are finished
    # This is automatically done when using one of the functions below to obtain the computation results
    # and is not necessary to call it explicitly in most cases
    cdef void llama_synchronize( llama_context * ctx)

    # Token logits obtained from the last call to llama_decode()
    # The logits for which llama_batch.logits[i] != 0 are stored contiguously
    # in the order they have appeared in the batch.
    # Rows: number of tokens for which llama_batch.logits[i] != 0
    # Cols: n_vocab
    cdef float * llama_get_logits( llama_context * ctx)

    # FIXME: should this be added
    # cdef int32_t llama_n_outputs( llama_context * ctx)

    # Logits for the ith token. For positive indices, Equivalent to:
    # llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    # Negative indicies can be used to access logits in reverse order, -1 is the last logit.
    # returns NULL for invalid ids.
    cdef float * llama_get_logits_ith( llama_context * ctx, int32_t i)

    # Get all output token embeddings.
    # when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    # the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    # in the order they have appeared in the batch.
    # shape: [n_outputs*n_embd]
    # Otherwise, returns NULL.
    cdef float * llama_get_embeddings( llama_context * ctx)

    # Get the embeddings for the ith token. For positive indices, Equivalent to:
    # llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    # Negative indicies can be used to access embeddings in reverse order, -1 is the last embedding.
    # returns NULL for invalid ids.
    cdef float * llama_get_embeddings_ith( llama_context * ctx, int32_t i)

    # Get the embeddings for a sequence id
    # Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    # when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
    # otherwise: float[n_embd] (1-dimensional)
    cdef float * llama_get_embeddings_seq( llama_context * ctx, llama_seq_id seq_id)


    # -------------------------------------------------------------------------
    # Vocab

    cdef const char * llama_vocab_get_text(const llama_vocab * vocab, llama_token token)

    cdef float llama_vocab_get_score(const llama_vocab * vocab, llama_token token)

    cdef llama_token_attr llama_vocab_get_attr(const llama_vocab * model, llama_token token)

    # Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    cdef bint llama_vocab_is_eog(const llama_vocab * vocab, llama_token token)

    # Identify if Token Id is a control token or a render-able token
    cdef bint llama_vocab_is_control(const llama_vocab * vocab, llama_token token)

    # Special tokens
    cdef llama_token llama_vocab_bos(const llama_vocab * vocab) # beginning-of-sentence
    cdef llama_token llama_vocab_eos(const llama_vocab * vocab) # end-of-sentence
    cdef llama_token llama_vocab_eot(const llama_vocab * vocab) # end-of-turn
    cdef llama_token llama_vocab_cls(const llama_vocab * vocab) # classification
    cdef llama_token llama_vocab_sep(const llama_vocab * vocab) # sentence separator
    cdef llama_token llama_vocab_nl (const llama_vocab * vocab) # next-line
    cdef llama_token llama_vocab_pad(const llama_vocab * vocab) # padding

    cdef bint llama_vocab_get_add_bos(const llama_vocab * vocab)
    cdef bint llama_vocab_get_add_eos(const llama_vocab * vocab)

    cdef llama_token llama_vocab_fim_pre(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_suf(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_mid(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_pad(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_rep(const llama_vocab * vocab)
    cdef llama_token llama_vocab_fim_sep(const llama_vocab * vocab)

    # -------------------------------------------------------------------------
    # Tokenization (The API is thread-safe)

    # @details Convert the provided text into tokens.
    # @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    # @return Returns the number of tokens on success, no more than n_tokens_max
    # @return Returns a negative number on failure - the number of tokens that would have been returned
    # @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    # @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    #                      as plaintext. Does not insert a leading space.
    cdef int32_t llama_tokenize(
        const  llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bint   add_special,
                            bint   parse_special)

    # Token Id -> Piece.
    # Uses the vocabulary in the provided context.
    # Does not write null terminator to the buffer.
    # User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    # @param special If true, special tokens are rendered in the output.
    cdef int32_t llama_token_to_piece(
              const  llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bint   special)


    # @details Convert the provided tokens into text (inverse of llama_tokenize()).
    # @param text The char pointer must be large enough to hold the resulting text.
    # @return Returns the number of chars/bytes on success, no more than text_len_max.
    # @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    # @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    # @param unparse_special If true, special tokens are rendered in the output.
    cdef int32_t llama_detokenize(
        const  llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bint   remove_special,
                            bint   unparse_special)


    # -------------------------------------------------------------------------
    # Chat templates
    #

    # Apply chat template. Inspired by hf apply_chat_template() on python.
    # Both "model" and "custom_template" are optional, but at least one is required. "custom_template" has higher precedence than "model"
    # NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https:#github.com/ggerganov/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    # @param tmpl A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.
    # @param chat Pointer to a list of multiple llama_chat_message
    # @param n_msg Number of llama_chat_message in this chat
    # @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    # @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    # @param length The size of the allocated buffer
    # @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    cdef int32_t llama_chat_apply_template(
                            const char * tmpl,
             const  llama_chat_message * chat,
                                size_t   n_msg,
                                  bint   add_ass,
                                  char * buf,
                               int32_t   length)

    # Get list of built-in chat templates
    cdef int32_t llama_chat_builtin_templates(const char ** output, size_t len)

    # -------------------------------------------------------------------------
    # Sampling API

    # Sample usage:
    #
    #    # prepare the sampling chain at the start
    #    auto sparams = llama_sampler_chain_default_params();
    #
    #    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    #
    #    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    #    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    #    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    #
    #    # typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    #    # this sampler will be responsible to select the actual token
    #    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    #
    #    ...
    #
    #    # decoding loop:
    #    while (...) {
    #        ...
    #
    #        llama_decode(ctx, batch);
    #
    #        # sample from the logits of the last token in the batch
    #        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    #
    #        # accepting the token updates the internal state of certain samplers (e.g. grammar, repetition, etc.)
    #        llama_sampler_accept(smpl, id);
    #        ...
    #    }
    #
    #    llama_sampler_free(smpl);
    #
    # TODO: In the future, llama_sampler will be utilized to offload the sampling to the backends (e.g. GPU).

    ctypedef void * llama_sampler_context_t

    # user code can implement the interface below in order to create custom llama_sampler
    ctypedef struct llama_sampler_i:
        const char *           (*name)  (const llama_sampler * smpl)                                 # can be NULL
        void                   (*accept)(      llama_sampler * smpl, llama_token token)              # can be NULL
        void                   (*apply) (      llama_sampler * smpl, llama_token_data_array * cur_p) # required
        void                   (*reset) (      llama_sampler * smpl)                                 # can be NULL
        llama_sampler *        (*clone) (const llama_sampler * smpl)                                 # can be NULL if ctx is NULL
        void                   (*free)  (      llama_sampler * smpl)

    ctypedef struct llama_sampler:
        const llama_sampler_i * iface
        llama_sampler_context_t ctx

    # mirror of llama_sampler_i:
    cdef llama_sampler *        llama_sampler_init  (const llama_sampler_i * iface, llama_sampler_context_t ctx)
    cdef const char *           llama_sampler_name  (const llama_sampler * smpl)
    cdef void                   llama_sampler_accept(      llama_sampler * smpl, llama_token token)
    cdef void                   llama_sampler_apply (      llama_sampler * smpl, llama_token_data_array * cur_p)
    cdef void                   llama_sampler_reset (      llama_sampler * smpl)
    cdef llama_sampler *        llama_sampler_clone (const llama_sampler * smpl)
    # important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    cdef void                   llama_sampler_free  (      llama_sampler * smpl)

    # llama_sampler_chain
    # a type of llama_sampler that can chain multiple samplers one after another

    cdef llama_sampler * llama_sampler_chain_init(llama_sampler_chain_params params)

    # important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    cdef void                   llama_sampler_chain_add(       llama_sampler * chain, llama_sampler * smpl)
    cdef llama_sampler *        llama_sampler_chain_get(const  llama_sampler * chain, int32_t i)
    cdef int                    llama_sampler_chain_n  (const  llama_sampler * chain)

    # after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    cdef llama_sampler * llama_sampler_chain_remove( llama_sampler * chain, int32_t i)

    # available samplers:

    cdef llama_sampler * llama_sampler_init_greedy()
    cdef llama_sampler * llama_sampler_init_dist(uint32_t seed)

    # DEPRECATED
    # @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    # cdef llama_sampler * llama_sampler_init_softmax()

    # @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751
    cdef llama_sampler * llama_sampler_init_top_k(int32_t k)

    # @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https:#arxiv.org/abs/1904.09751
    cdef llama_sampler * llama_sampler_init_top_p (float p, size_t min_keep)

    # @details Minimum P sampling as described in https:#github.com/ggerganov/llama.cpp/pull/3841
    cdef llama_sampler * llama_sampler_init_min_p (float p, size_t min_keep)

    # @details Locally Typical Sampling implementation described in the paper https:#arxiv.org/abs/2202.00666.
    cdef llama_sampler * llama_sampler_init_typical (float p, size_t min_keep)

    # @details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    cdef llama_sampler * llama_sampler_init_temp (float t)

    # @details Dynamic temperature implementation described in the paper https:#arxiv.org/abs/2309.02772.
    cdef llama_sampler * llama_sampler_init_temp_ext (float t, float delta, float exponent)

    # @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    cdef llama_sampler * llama_sampler_init_xtc (float p, float t, size_t min_keep, uint32_t seed)

    # @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    cdef llama_sampler * llama_sampler_init_top_n_sigma(float n)

    # @details Mirostat 1.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.
    # @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    # @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    # @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    # @pam m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    # @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    cdef llama_sampler * llama_sampler_init_mirostat(
                 int32_t   n_vocab,
                uint32_t   seed,
                   float   tau,
                   float   eta,
                 int32_t   m)

    # @details Mirostat 2.0 algorithm described in the paper https:#arxiv.org/abs/2007.14966. Uses tokens instead of words.
    # @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    # @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    # @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    # @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    cdef llama_sampler * llama_sampler_init_mirostat_v2(
                                uint32_t   seed,
                                   float   tau,
                                   float   eta)

    # @details Intializes a GBNF grammar, see grammars/README.md for details.
    # @param vocab The vocabulary that this grammar will be used with.
    # @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    # @param grammar_root The name of the start symbol for the grammar.
    cdef llama_sampler * llama_sampler_init_grammar(
                          const llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root)

    # @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    # @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    # @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    cdef llama_sampler * llama_sampler_init_grammar_lazy_patterns(
               const llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens)


    # NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    cdef llama_sampler * llama_sampler_init_penalties(
                             int32_t   penalty_last_n,   # last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   # 1.0 = disabled
                               float   penalty_freq,     # 0.0 = disabled
                               float   penalty_present)  # 0.0 = disabled

    # @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    cdef llama_sampler * llama_sampler_init_dry(
                   const llama_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers)

    cdef llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias)

    # this sampler is meant to be used for fill-in-the-middle infilling
    # it's supposed to be used after top_k + top_p sampling
    #
    # 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    # 2. combine probs of tokens that have the same prefix
    #
    # example:
    #
    # - before:
    #   "hel":   0.5
    #   "hell":  0.2
    #   "hello": 0.1
    #   "dummy": 0.1
    #
    # - after:
    #   "hel":   0.8
    #   "dummy": 0.1
    #
    # 3. discard non-EOG tokens with low prob
    # 4. if no tokens are left -> pick EOT
    #
    cdef llama_sampler * llama_sampler_init_infill(const llama_vocab * vocab)


    # Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    cdef uint32_t llama_sampler_get_seed(const llama_sampler * smpl)

    # Sample and accept a token from the idx-th output of the last evaluation
    #
    # Shorthand for:
    #
    #    const auto * logits = llama_get_logits_ith(ctx, idx)
    #    llama_token_data_array cur_p = { ... init from logits ... }
    #    llama_sampler_apply(smpl, &cur_p)
    #    return cur_p.data[cur_p.selected].id
    #
    # At this point, this is mostly a convenience function.

    cdef llama_token llama_sampler_sample(llama_sampler * smpl, llama_context * ctx, int32_t idx)

    # TODO: extend in the future
    # void llama_decode_with_sampler(llama_context * ctx, llama_sampler * smpl, llama_batch batch, ...)

    # -------------------------------------------------------------------------
    # Model split

    # @details Build a split GGUF final path for this chunk.
    #          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    #  Returns the split_path length.
    cdef int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count)

    # @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    #          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    #  Returns the split_prefix length.
    cdef int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count)

    # Print system information
    cdef const char * llama_print_system_info()

    # Set callback for all future logging events.
    # If this is not called, or NULL is supplied, everything is output on stderr.
    cdef void llama_log_set(ggml_log_callback log_callback, void * user_data)

    # -------------------------------------------------------------------------
    # Performance utils

    # NOTE: Used by llama.cpp examples, avoid using in third-party apps. Instead, do your own performance measurements.
    #

    ctypedef struct llama_perf_context_data:
        double t_start_ms
        double t_load_ms
        double t_p_eval_ms
        double t_eval_ms

        int32_t n_p_eval
        int32_t n_eval

    ctypedef struct llama_perf_sampler_data:
        double t_sample_ms

        int32_t n_sample

    cdef llama_perf_context_data llama_perf_context(const llama_context * ctx)
    cdef void llama_perf_context_print(const llama_context * ctx)
    cdef void llama_perf_context_reset(      llama_context * ctx)

    # NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    cdef llama_perf_sampler_data llama_perf_sampler(const llama_sampler * chain)
    cdef void llama_perf_sampler_print(const llama_sampler * chain)
    cdef void llama_perf_sampler_reset(      llama_sampler * chain)

    #
    # training
    #

    # function that returns whether or not a given tensor contains trainable parameters
    ctypedef bint (*llama_opt_param_filter)(const ggml_tensor * tensor, void * userdata)

    # always returns true
    cdef bint llama_opt_param_filter_all(const ggml_tensor * tensor, void * userdata)

    ctypedef struct llama_opt_params:
        uint32_t n_ctx_train # assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter # callback for determining which tensors contain trainable parameters
        void * param_filter_ud              # userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars # callback for calculating optimizer parameters
        void * get_opt_pars_ud                     # userdata for calculating optimizer parameters

    cdef void llama_opt_init(llama_context * lctx, llama_model * model, llama_opt_params lopt_params)

    cdef void llama_opt_epoch(
            llama_context             * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval)


    # cdef void llama_perf_dump_yaml(FILE * stream, const llama_context * ctx)


#------------------------------------------------------------------------------
# llama-cpp.h

cdef extern from "llama-cpp.h":

    ctypedef struct llama_model_deleter: pass
    ctypedef struct llama_context_deleter: pass
    ctypedef struct llama_sampler_deleter: pass
    ctypedef struct llama_adapter_lora_deleter: pass

    ctypedef unique_ptr[llama_model, llama_model_deleter] llama_model_ptr
    ctypedef unique_ptr[llama_context, llama_context_deleter] llama_context_ptr
    ctypedef unique_ptr[llama_sampler, llama_sampler_deleter] llama_sampler_ptr
    ctypedef unique_ptr[llama_adapter_lora, llama_adapter_lora_deleter] llama_adapter_lora_ptr


#------------------------------------------------------------------------------
# common.h

cdef extern from "common.h":

    ctypedef std_vector[llama_token] llama_tokens

    ctypedef struct common_adapter_lora_info:
        std_string path
        float scale
        llama_adapter_lora * ptr

    ctypedef struct common_control_vector_load_info: pass

    # -------------------------------------------------------------------------
    # Build info

    cdef int LLAMA_BUILD_NUMBER
    cdef const char * LLAMA_COMMIT
    cdef const char * LLAMA_COMPILER
    cdef const char * LLAMA_BUILD_TARGET

    # -------------------------------------------------------------------------
    # CPU utils

    ctypedef struct cpu_params:
        int      n_threads
        bint     cpumask[GGML_MAX_N_THREADS] # CPU affinity mask.
        bint     mask_valid             # Default: any CPU
        ggml_sched_priority  priority   # Scheduling prio : (0 - normal, 1 - medium, 2 - high, 3 - realtime)
        bint     strict_cpu             # Use strict CPU placement
        uint32_t poll                   # Polling (busywait) level (0 - no polling, 100 - mostly polling)


    cdef int32_t cpu_get_num_physical_cores()
    cdef int32_t cpu_get_num_math()

    # -------------------------------------------------------------------------
    # Common params

    cdef enum llama_example:
        LLAMA_EXAMPLE_COMMON
        LLAMA_EXAMPLE_SPECULATIVE
        LLAMA_EXAMPLE_MAIN
        LLAMA_EXAMPLE_INFILL
        LLAMA_EXAMPLE_EMBEDDING
        LLAMA_EXAMPLE_PERPLEXITY
        LLAMA_EXAMPLE_RETRIEVAL
        LLAMA_EXAMPLE_PASSKEY
        LLAMA_EXAMPLE_IMATRIX
        LLAMA_EXAMPLE_BENCH
        LLAMA_EXAMPLE_SERVER
        LLAMA_EXAMPLE_CVECTOR_GENERATOR
        LLAMA_EXAMPLE_EXPORT_LORA
        LLAMA_EXAMPLE_LLAVA
        LLAMA_EXAMPLE_LOOKUP
        LLAMA_EXAMPLE_PARALLEL
        LLAMA_EXAMPLE_TTS
        LLAMA_EXAMPLE_COUNT

    cdef enum common_sampler_type:
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

    # dimensionality reduction methods, used by cvector-generator
    cdef enum dimre_method:
        DIMRE_METHOD_PCA
        DIMRE_METHOD_MEAN

    cdef enum common_conversation_mode:
        COMMON_CONVERSATION_MODE_DISABLED = 0
        COMMON_CONVERSATION_MODE_ENABLED  = 1
        COMMON_CONVERSATION_MODE_AUTO     = 2

    cdef enum common_grammar_trigger_type:
        COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN
        COMMON_GRAMMAR_TRIGGER_TYPE_WORD
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN
        COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_START

    ctypedef struct common_grammar_trigger:
        common_grammar_trigger_type type
        std_string value
        llama_token token


    # sampler parameters
    ctypedef struct common_params_sampling:
        uint32_t seed  # the seed used to initialize llama_sampler

        int32_t n_prev                     # number of previous tokens to remember
        int32_t n_probs                    # if greater than 0, output the probabilities of top n_probs tokens.
        int32_t min_keep                   # 0 = disabled, otherwise samplers should return at least min_keep tokens
        int32_t top_k                      # <= 0 to use vocab size
        float   top_p                      # 1.0 = disabled
        float   min_p                      # 0.0 = disabled
        float   xtc_probability            # 0.0 = disabled
        float   xtc_threshold              # > 0.5 disables XTC
        # float   tfs_z                    # 1.0 = disabled
        float   typ_p                      # typical_p, 1.0 = disabled
        float   temp                       # <= 0.0 to sample greedily, 0.0 to not output probabilities
        float   dynatemp_range             # 0.0 = disabled
        float   dynatemp_exponent          # controls how entropy maps to temperature in dynamic temperature sampler
        int32_t penalty_last_n             # last n tokens to penalize (0 = disable penalty, -1 = context size)
        float   penalty_repeat             # 1.0 = disabled
        float   penalty_freq               # 0.0 = disabled
        float   penalty_present            # 0.0 = disabled
        float   dry_multiplier             # 0.0 = disabled; DRY repetition penalty for tokens extending repetition:
        float   dry_base                   # 0.0 = disabled; multiplier * base ^ (length of sequence before token - allowed length)
        int32_t dry_allowed_length         # tokens extending repetitions beyond this receive penalty
        int32_t dry_penalty_last_n         # how many tokens to scan for repetitions (0 = disable penalty, -1 = context size)
        int32_t mirostat                   # 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
        float   mirostat_tau               # target entropy
        float   mirostat_eta               # learning rate
        bint    ignore_eos                 # ignore end-of-sentence
        bint    no_perf                    # disable performance metrics
        bint    timing_per_token

        std_vector[std_string] dry_sequence_breakers

        std_vector[common_sampler_type] samplers

        std_string grammar # optional BNF-like grammar to constrain sampling
        bint grammar_lazy
        std_vector[common_grammar_trigger] grammar_triggers
        std_set[llama_token] preserved_tokens

        std_vector[llama_logit_bias] logit_bias # logit biases to apply

        # print the parameters into a string
        # std_string print() const


    ctypedef struct common_params_model:
        std_string path         # model local path                                           // NOLINT
        std_string url          # model url to download                                      // NOLINT
        std_string hf_repo      # HF repo                                                    // NOLINT
        std_string hf_file      # HF file                                                    // NOLINT

    ctypedef struct common_params_speculative:
        std_vector[ggml_backend_dev_t] devices # devices to use for offloading
        int32_t n_ctx           # draft context size
        int32_t n_max           # maximum number of tokens to draft during speculative decoding
        int32_t n_min           # minimum number of draft tokens to use for speculative decoding
        int32_t n_gpu_layers    # number of layers to store in VRAM for the draft model (-1 - use default)
        float   p_split         # speculative decoding split probability
        float   p_min           # minimum speculative decoding probability (greedy)

        cpu_params cpuparams
        cpu_params cpuparams_batch

        common_params_model model       # draft model for speculative decoding


    ctypedef struct common_params_vocoder:
        common_params_model model

        std_string speaker_file     # speaker file path
        bint use_guide_tokens       # enable guide tokens to improve TTS accuracy

    cdef enum common_reasoning_format:
        COMMON_REASONING_FORMAT_NONE
        COMMON_REASONING_FORMAT_DEEPSEEK # Extract thinking tag contents and return as `message.reasoning_content`

    ctypedef struct common_params:
        int32_t n_predict          # new tokens to predict
        int32_t n_ctx              # context size
        int32_t n_batch            # logical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_ubatch           # physical batch size for prompt processing (must be >=32 to use BLAS)
        int32_t n_keep             # number of tokens to keep from initial prompt
        int32_t n_chunks           # max number of chunks to process (-1 = unlimited)
        int32_t n_parallel         # number of parallel sequences to decode
        int32_t n_sequences        # number of sequences to decode
        int32_t grp_attn_n         # group-attention factor
        int32_t grp_attn_w         # group-attention width
        int32_t n_print            # print token count every n tokens (-1 = disabled)
        float   rope_freq_base     # RoPE base frequency
        float   rope_freq_scale    # RoPE frequency scaling factor
        float   yarn_ext_factor    # YaRN extrapolation mix factor
        float   yarn_attn_factor   # YaRN magnitude scaling factor
        float   yarn_beta_fast     # YaRN low correction dim
        float   yarn_beta_slow     # YaRN high correction dim
        int32_t yarn_orig_ctx      # YaRN original context length
        float   defrag_thold       # KV cache defragmentation threshold

        std_vector[ggml_backend_dev_t] devices # devices to use for offloading
        
        int32_t n_gpu_layers       # number of layers to store in VRAM (-1 - use default)
        int32_t main_gpu           # the GPU that is used for scratch and small tensors
        float   tensor_split[128]  # how split tensors should be distributed across GPUs
        
        llama_split_mode split_mode # how to split the model across GPUs

        cpu_params cpuparams
        cpu_params cpuparams_batch

        ggml_backend_sched_eval_callback cb_eval
        void * cb_eval_user_data

        ggml_numa_strategy numa

        llama_rope_scaling_type rope_scaling_type
        llama_pooling_type      pooling_type       # pooling type for embeddings
        llama_attention_type    attention_type     # attention type for embeddings

        common_params_sampling    sampling
        common_params_speculative speculative
        common_params_vocoder     vocoder

        common_params_model model 

        std_string model_alias          # model alias
        std_string hf_token             # HF token
        std_string prompt               #
        std_string system_prompt
        std_string prompt_file          # store the external prompt file name
        std_string path_prompt_cache    # path to file for saving/loading prompt eval state
        std_string input_prefix         # string to prefix user inputs with
        std_string input_suffix         # string to suffix user inputs with
        std_string lookup_cache_static  # path of static ngram cache file for lookup decoding
        std_string lookup_cache_dynamic # path of dynamic ngram cache file for lookup decoding
        std_string logits_file          # file for saving *all* logits

        std_vector[std_string] in_files     # all input files
        std_vector[std_string] antiprompt   # strings upon which more user input is prompted (a.k.a. reverse prompts)
        std_vector[llama_model_kv_override] kv_overrides
        # std_vector[llama_model_tensor_buft_override] tensor_buft_overrides

        bint lora_init_without_apply # only load lora to memory, but do not apply it to ctx (user can manually apply lora later using llama_adapter_lora_apply)
        # vector[common_adapter_lora_info] lora_adapters # lora adapter path with user defined scale

        # vector[common_control_vector_load_info] control_vectors # control vector with user defined scale

        int32_t verbosity
        int32_t control_vector_layer_start # layer range for control vector
        int32_t control_vector_layer_end   # layer range for control vector

        int32_t ppl_stride          # stride for perplexity calculations. If left at 0, the pre-existing approach will be used.
        int32_t ppl_output_type     # = 0 -> ppl output is as usual, = 1 -> ppl output is num_tokens, ppl, one per line

        bint   hellaswag            # compute HellaSwag score over random tasks from datafile supplied in prompt
        size_t hellaswag_tasks      # number of tasks to use when computing the HellaSwag score

        bint   winogrande           # compute Winogrande score over random tasks from datafile supplied in prompt
        size_t winogrande_tasks     # number of tasks to use when computing the Winogrande score. If 0, all tasks will be computed

        bint   multiple_choice      # compute TruthfulQA score over random tasks from datafile supplied in prompt
        size_t multiple_choice_tasks # number of tasks to use when computing the TruthfulQA score. If 0, all tasks will be computed

        bint   kl_divergence        # compute KL divergence

        # std::function<void(int, char **)> print_usage # print example-specific usage and example
        bint usage                  # print usage
        bint use_color              # use color to distinguish generations and inputs
        bint special                # enable special token output
        bint interactive            # interactive mode
        bint interactive_first      # wait for user input immediately
        bint conversation           # conversation mode (does not print special tokens and suffix/prefix)
        bint prompt_cache_all       # save user input and generations to prompt cache
        bint prompt_cache_ro        # open the prompt cache read-only and do not update it

        bint escape                 # escape "\n", "\r", "\t", "\'", "\"", and "\\"
        bint multiline_input        # reverse the usage of `\`
        bint simple_io              # improves compatibility with subprocesses and limited consoles
        bint cont_batching          # insert new sequences for decoding on-the-fly
        bint flash_attn             # flash attention
        bint no_perf                # disable performance metric
        bint ctx_shift              # context shift on inifinite text generation

        bint input_prefix_bos       # prefix BOS to user inputs, preceding input_prefix
        bint logits_all             # return logits for all tokens in the batch
        bint use_mmap               # use mmap for faster loads
        bint use_mlock              # use mlock to keep model in memory
        bint verbose_prompt         # print prompt tokens before generation
        bint display_prompt         # print prompt before generation
        bint infill                 # use infill mode
        bint dump_kv_cache          # dump the KV cache contents for debugging purposes
        bint no_kv_offload          # disable KV offloading
        bint warmup                 # warmup run
        bint check_tensors          # validate tensor data

        ggml_type cache_type_k      # KV cache data type for the K
        ggml_type cache_type_v      # KV cache data type for the V

        # multimodal models (see examples/llava)
        std_string mmproj           # path to multimodal projector
        std_vector[std_string] image # path to image file(s)

        # embedding
        bint embedding              # get only sentence embedding
        int32_t embd_normalize      # normalisation for embeddings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
        std_string embd_out         # empty = default, "array" = [[],[]...], "json" = openai style, "json+" = same "json" + cosine similarity matrix
        std_string embd_sep         # separator of embeddings
        bint reranking              # enable reranking support on server

        # server params
        int32_t port                # server listens on this network port
        int32_t timeout_read        # http read timeout in seconds
        int32_t timeout_write       # http write timeout in seconds
        int32_t n_threads_http      # number of threads to process HTTP requests (TODO: support threadpool)
        int32_t n_cache_reuse       # min chunk size to reuse from the cache via KV shifting

        std_string hostname
        std_string public_path
        std_string chat_template
        bint enable_chat_template

        std_vector[std_string] api_keys

        std_string ssl_file_key
        std_string ssl_file_cert

        bint webui
        bint endpoint_slots
        bint endpoint_props
        bint endpoint_metrics

        bint log_json

        std_string slot_save_path

        float slot_prompt_similarity

        # batched-bench params
        bint is_pp_shared

        std_vector[int32_t] n_pp
        std_vector[int32_t] n_tg
        std_vector[int32_t] n_pl

        # retrieval params
        std_vector[std_string] context_files # context files to embed

        int32_t chunk_size      # chunk size for context embedding

        std_string chunk_separator # chunk separator for context embedding

        # passkey params
        int32_t n_junk      # number of times to repeat the junk text
        int32_t i_pos       # position of the passkey in the junk text

        # imatrix params
        std_string out_file # save the resulting imatrix to this file

        int32_t n_out_freq       # output the imatrix every n_out_freq iterations
        int32_t n_save_freq      # save the imatrix every n_save_freq iterations
        int32_t i_chunk          # start processing from this chunk

        bint process_output      # collect data for the output tensor
        bint compute_ppl         # whether to compute perplexity

        # cvector-generator params
        int n_pca_batch
        int n_pca_iterations
        # dimre_method cvector_dimre_method
        std_string cvector_outfile
        std_string cvector_positive_file
        std_string cvector_negative_file

        bint spm_infill

        std_string lora_outfile

        # batched-bench params
        bint batched_bench_output_jsonl

    # call once at the start of a program if it uses libcommon
    # initializes the logging system and prints info about the build
    cdef void common_init()

    cdef std_string common_params_get_system_info(const common_params & params)

    # cdef bint parse_cpu_range(const std_string & range, bool(&boolmask)[GGML_MAX_N_THREADS])
    # cdef bint parse_cpu_mask(const std_string & mask, bool(&boolmask)[GGML_MAX_N_THREADS])
    cdef void postprocess_cpu_params(cpu_params & cpuparams, const cpu_params * role_model)
    cdef bint set_process_priority(ggml_sched_priority prio)

    cdef llama_model * common_load_model_from_url(
        const std_string & model_url,
        const std_string & local_path,
        const std_string & hf_token,
        const llama_model_params & params)

    cdef llama_model * common_load_model_from_hf(
        const std_string & repo,
        const std_string & remote_path,
        const std_string & local_path,
        const std_string & hf_token,
        const llama_model_params & params)

    # Token utils

    # longest common prefix
    cdef size_t common_lcp(const llama_tokens & a, const llama_tokens & b)

    # longet common subsequence
    cdef size_t common_lcs(const llama_tokens & a, const llama_tokens & b)


    # -------------------------------------------------------------------------
    # String utils

    cdef std_string string_from(bint value)
    cdef std_string string_from(const std_vector[int] & values)
    cdef std_string string_from(const llama_context * ctx, const std_vector[llama_token] & tokens)
    cdef std_string string_from(const llama_context * ctx, const llama_batch & batch)

    # -------------------------------------------------------------------------
    # Model utils

    # note: defines object's lifetime
    ctypedef struct common_init_result:
        llama_model_ptr model
        llama_context_ptr context
        std_vector[llama_adapter_lora_ptr] lora

    cdef common_init_result common_init_from_params(common_params & params)

    cdef llama_model_params common_model_params_to_llama(common_params & params)
    cdef llama_context_params common_context_params_to_llama(const common_params & params)
    cdef ggml_threadpool_params ggml_threadpool_params_from_cpu_params(const cpu_params & params);

    cdef llama_model * common_load_model_from_url(const std_string & model_url, const std_string & local_path, const std_string & hf_token, const llama_model_params & params)
    cdef llama_model * common_load_model_from_hf(const std_string & repo, const std_string & remote_path, const std_string & local_path, const std_string & hf_token, const llama_model_params & params)

    # clear LoRA adapters from context, then apply new list of adapters
    cdef void common_lora_adapters_apply(llama_context * ctx, std_vector[common_adapter_lora_info] & lora)

    # -------------------------------------------------------------------------
    # Batch utils

    cdef void common_batch_add(llama_batch & batch, llama_token id, llama_pos pos, const std_vector[llama_seq_id] & seq_ids, bint logits)

    cdef void common_batch_clear(llama_batch & batch)

    # -------------------------------------------------------------------------
    # Token utils

    # longest common prefix
    cdef size_t common_lcp(const llama_tokens & a, const llama_tokens & b)

    # longet common subsequence
    cdef size_t common_lcs(const llama_tokens & a, const llama_tokens & b)


    # -------------------------------------------------------------------------
    # Vocab utils

    cdef std_vector[llama_token] common_tokenize(const llama_context * ctx, const std_string & text, bint add_special, bint parse_special)

    cdef std_vector[llama_token] common_tokenize(const llama_model * model, const std_string & text, bint add_special, bint parse_special)

    # tokenizes a token into a piece, optionally renders special/control tokens
    # should work similar to Python's `tokenizer.id_to_piece`

    # tokenizes a token into a piece, optionally renders special/control tokens
    # should work similar to Python's `tokenizer.id_to_piece`
    cdef std_string common_token_to_piece (const llama_context * ctx, llama_token token, bint special)

    # detokenizes a vector of tokens into a string
    # should work similar to Python's `tokenizer.decode`
    # optionally renders special/control tokens
    cdef std_string common_detokenize(llama_context * ctx, const std_vector[llama_token] & tokens, bint special)

    # -------------------------------------------------------------------------
    # Chat template utils

    # same with llama_chat_message, but uses std::string
    ctypedef struct common_chat_msg:
        std_string role
        std_string content

    #  Get the built-in chat template for the model. Return empty string if not present.
    cdef std_string common_get_builtin_chat_template(const llama_model * model)

    # Check if the template supplied via "--chat-template" is supported or not. Returns true if it's valid
    cdef bint common_chat_verify_template(const std_string & tmpl)

    # CPP wrapper for common_chat_apply_template
    # If the built-in template is not supported, we default to chatml
    # If the custom "tmpl" is not supported, we throw an error
    cdef std_string common_chat_apply_template(const llama_model * model, const std_string & tmpl, const std_vector[common_chat_msg] & chat, bint add_ass)

    # Format single message, while taking into account the position of that message in chat history
    cdef std_string common_chat_format_single(const llama_model * model, const std_string & tmpl, const std_vector[common_chat_msg] & past_msg, const common_chat_msg & new_msg, bint add_ass)

    # # Returns an example of formatted chat
    cdef std_string common_chat_format_example(const llama_model * model, const std_string & tmpl)

    # -------------------------------------------------------------------------
    # KV cache utils

    # # Dump the KV cache view with the number of sequences per cell.
    cdef void common_kv_cache_dump_view(const llama_kv_cache_view & view, int row_size)

    # # Dump the KV cache view showing individual sequences in each cell (long output).
    cdef void common_kv_cache_dump_view_seqs(const llama_kv_cache_view & view, int row_size)

    # -------------------------------------------------------------------------
    # Embedding utils

    # TODO: repace embd_norm with an enum
    cdef void common_embd_normalize(const float * inp, float * out, int n, int embd_norm)
    cdef float common_embd_similarity_cos(const float * embd1, const float * embd2, int n)

    # -------------------------------------------------------------------------
    # Control vector utils

    ctypedef struct common_control_vector_data:
        int n_embd
        # stores data for layers [1, n_layer] where n_layer = data.size() / n_embd
        std_vector[float] data

    ctypedef struct common_control_vector_load_info:
        float strength
        std_string fname

    # Load control vectors, scale each by strength, and add them together.
    # On error, returns {-1, empty}
    cdef common_control_vector_data common_control_vector_load(const std_vector[common_control_vector_load_info] & load_infos)

    # -------------------------------------------------------------------------
    # Split Utils

    # ..


    # -------------------------------------------------------------------------
    # YAML utils

    # ..

#------------------------------------------------------------------------------
# sampling.h

cdef extern from "sampling.h": # optional llama_sampler extensions
    ctypedef struct common_sampler: pass # opaque

    # llama_sampler API overloads

    cdef common_sampler * common_sampler_init(const llama_model * model, const common_params_sampling & params)

    void common_sampler_free(common_sampler * gsmpl);

    # if accept_grammar is true, the token is accepted both by the sampling chain and the grammar
    void common_sampler_accept(common_sampler * gsmpl, llama_token token, bint accept_grammar)
    void common_sampler_reset (common_sampler * gsmpl)
    common_sampler * common_sampler_clone (common_sampler * gsmpl)

    # arguments can be nullptr to skip printing
    void common_perf_print(const llama_context * ctx, const common_sampler * gsmpl)

    # extended sampling implementation:
    #
    # - set logits
    # - apply the configured sampler chain
    # - check if the token fits the grammar (if any)
    # - if not: resample by first applying the grammar constraints and then sampling again (slower path)
    #
    # if grammar_first is true, the grammar is applied before the samplers (slower)
    # useful in cases where all the resulting candidates (not just the sampled one) must fit the grammar
    #
    llama_token common_sampler_sample(common_sampler * gsmpl, llama_context * ctx, int idx, bint grammar_first)

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
    #
    std_vector[llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama_context * ctx, const std_vector[int] & idxs, const llama_tokens & draft, bint grammar_first)

    # assume idxs == [ 0, 1, 2, ..., draft.size() ]
    std_vector[llama_token] common_sampler_sample_and_accept_n(common_sampler * gsmpl, llama_context * ctx, const llama_tokens & draft, bint grammar_first)

    uint32_t common_sampler_get_seed(const common_sampler * gsmpl)

    # helpers

    # access the internal list of current candidate tokens
    llama_token_data_array * common_sampler_get_candidates(common_sampler * gsmpl)

    # get the last accepted token
    llama_token common_sampler_last(const common_sampler * gsmpl)

    # print the sampler chain into a string
    std_string common_sampler_print(const common_sampler * gsmpl)

    # get a string representation of the last accepted tokens
    std_string common_sampler_prev_str(common_sampler * gsmpl, llama_context * ctx, int n)

    char common_sampler_type_to_chr(common_sampler_type cnstr)
    std_string common_sampler_type_to_str(common_sampler_type cnstr)

    std_vector[common_sampler_type] common_sampler_types_from_names(const std_vector[std_string] & names, bint allow_alt_names)
    std_vector[common_sampler_type] common_sampler_types_from_chars(const std_string & chars)

#------------------------------------------------------------------------------
# log.h

cdef extern from "log.h":

    cdef void common_log_set_verbosity_thold(int verbosity)

#------------------------------------------------------------------------------
# arg.h

cdef extern from "arg.h":

    ctypedef struct common_arg:
        std_set[llama_example] examples
        std_set[llama_example] excludes
        std_vector[const char *] args
        const char * value_hint_2   # help text or example for arg value
        const char * value_hint_2   # for second arg value
        const char * env
        std_string help
        bint is_sparam              # is current arg a sampling param?
        void (*handler_void)   (common_params & params)
        void (*handler_string) (common_params & params, const std_string &)
        void (*handler_str_str)(common_params & params, const std_string &, const std_string &)
        void (*handler_int)    (common_params & params, int)

    ctypedef struct common_params_context:
        llama_example ex
        # common_params& params
        common_params params
        std_vector[common_arg] options
        void(*print_usage)(int, char **)

    ctypedef void(*print_usage)(int, char **)

    # parse input arguments from CLI
    # if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
    cdef bint common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, print_usage callback)

    # function to be used by test-arg-parser
    cdef common_params_context common_params_parser_init(common_params & params, llama_example ex, print_usage callback)

# #------------------------------------------------------------------------------
# cdef extern from "clip.h":

#     # added this (it's in clip.cpp) ---------

#     # RGB uint8 image
#     ctypedef struct clip_image_u8:
#         int nx
#         int ny
#         std_vector[uint8_t] buf

#     # RGB float32 image (NHWC)
#     # Memory layout: RGBRGBRGB...
#     ctypedef struct clip_image_f32:
#         int nx
#         int ny

#         std_vector[float] buf

#     # end (additions) ------------------------

#     ctypedef struct clip_ctx: pass

#     ctypedef struct clip_image_size:
#         int width
#         int height

#     ctypedef struct clip_image_u8_batch:
#         clip_image_u8 * data
#         size_t size

#     ctypedef struct clip_image_f32_batch:
#         clip_image_f32 * data
#         size_t size

#     cdef clip_ctx * clip_model_load    (const char * fname, int verbosity)
#     cdef clip_ctx * clip_model_load_cpu(const char * fname, int verbosity)

#     cdef void clip_free(clip_ctx * ctx)

#     cdef size_t clip_embd_nbytes(const clip_ctx * ctx)

#     cdef int32_t get_clip_image_size "get_clip_image_size" (const clip_ctx * ctx)
#     cdef int32_t clip_patch_size (const clip_ctx * ctx)
#     cdef int32_t clip_hidden_size(const clip_ctx * ctx)

#     # TODO: should be enum, not string
#     cdef const char * clip_patch_merge_type(const clip_ctx * ctx)

#     cdef const int32_t * clip_image_grid(const clip_ctx * ctx)

#     cdef int clip_n_patches    (const clip_ctx * ctx)
#     cdef int clip_n_mmproj_embd(const clip_ctx * ctx)

#     cdef int clip_uhd_num_image_embeds_col(clip_ctx * ctx_clip)
#     cdef void clip_add_load_image_size(clip_ctx * ctx_clip, clip_image_size * load_image_size)

#     cdef clip_image_size * clip_image_size_init()
#     cdef clip_image_u8  * clip_image_u8_init ()
#     cdef clip_image_f32 * clip_image_f32_init()

#     cdef void clip_image_u8_free (clip_image_u8  * img)
#     cdef void clip_image_f32_free(clip_image_f32 * img)
#     cdef void clip_image_u8_batch_free (clip_image_u8_batch  * batch)
#     cdef void clip_image_f32_batch_free(clip_image_f32_batch * batch)

#     cdef bint clip_image_load_from_file(const char * fname, clip_image_u8 * img)

#     # interpret bytes as an image file with length bytes_length, and use the result to populate img
#     cdef bint clip_image_load_from_bytes(const unsigned char * bytes, size_t bytes_length, clip_image_u8 * img)

#     # preprocess img and store the result in res_imgs, pad_to_square may be overridden to false depending on model configuration
#     cdef bint clip_image_preprocess(clip_ctx * ctx, const clip_image_u8 * img, clip_image_f32_batch * res_imgs )

#     cdef ggml_tensor * clip_get_newline_tensor(const clip_ctx * ctx)

#     cdef bint clip_image_encode      (clip_ctx * ctx, int n_threads, clip_image_f32 * img, float * vec)
#     cdef bint clip_image_batch_encode(clip_ctx * ctx, int n_threads, const clip_image_f32_batch * imgs, float * vec)

#     cdef bint clip_model_quantize(const char * fname_inp, const char * fname_out, int itype)

#     cdef int clip_is_minicpmv(const clip_ctx * ctx)


# #------------------------------------------------------------------------------
# # llava.h

# cdef extern from "llava.h":

#     ctypedef struct clip_ctx: pass

#     ctypedef struct llava_image_embed:
#         float * embed
#         int n_image_pos

#     # sanity check for clip <-> llava embed size match
#     cdef bint llava_validate_embed_size(const llama_context * ctx_llama, const clip_ctx * ctx_clip)

#     cdef bint llava_image_embed_make_with_clip_img(clip_ctx * ctx_clip, int n_threads, const clip_image_u8 * img, float ** image_embd_out, int * n_img_pos_out)

#     # build an image embed from image file bytes
#     cdef llava_image_embed * llava_image_embed_make_with_bytes(clip_ctx * ctx_clip, int n_threads, const unsigned char * image_bytes, int image_bytes_length)

#     # build an image embed from a path to an image filename
#     cdef llava_image_embed * llava_image_embed_make_with_filename(clip_ctx * ctx_clip, int n_threads, const char * image_path)

#     # free an embedding made with llava_image_embed_make_*
#     cdef void llava_image_embed_free(llava_image_embed * embed)

#     # write the image represented by embed into the llama context with batch size n_batch, starting at context pos n_past.
#     # on completion, n_past points to the next position in the context after the image embed.
#     cdef bint llava_eval_image_embed(llama_context * ctx_llama, const llava_image_embed * embed, int n_batch, int * n_past)


#------------------------------------------------------------------------------
# llamalib.h

cdef extern from "llamalib.h":
    cdef std_string simple_prompt(const std_string model_path, const std_string prompt, const int n_predict, const int n_ctx, bint disable_log, int n_threads)
