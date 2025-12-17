# distutils: language=c++

from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t, uint64_t, uint8_t
from libc.stdio cimport FILE
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.memory cimport unique_ptr as std_unique_ptr
from libcpp.set cimport set as std_set
from libcpp.functional cimport function as std_function

cimport ggml

#------------------------------------------------------------------------------
# log.h

cdef extern from "log.h":

    # Log level constants
    cdef int LOG_LEVEL_DEBUG   # 4
    cdef int LOG_LEVEL_INFO    # 3
    cdef int LOG_LEVEL_WARN    # 2
    cdef int LOG_LEVEL_ERROR   # 1
    cdef int LOG_LEVEL_OUTPUT  # 0 - output data from tools

    # Default log levels (now map to LOG_LEVEL_* constants)
    cdef int LOG_DEFAULT_DEBUG  # LOG_LEVEL_DEBUG
    cdef int LOG_DEFAULT_LLAMA  # LOG_LEVEL_INFO

    ctypedef enum log_colors:
        LOG_COLORS_AUTO     = -1
        LOG_COLORS_DISABLED = 0
        LOG_COLORS_ENABLED  = 1

    # External variable for log verbosity threshold
    cdef extern int common_log_verbosity_thold

    # Common log structure
    ctypedef struct common_log:
        pass

    # Function declarations
    cdef void common_log_set_verbosity_thold(int verbosity)
    cdef void common_log_default_callback(ggml.ggml_log_level level, const char * text, void * user_data)

    cdef common_log * common_log_init()
    cdef common_log * common_log_main()
    cdef void common_log_pause(common_log * log)
    cdef void common_log_resume(common_log * log)
    cdef void common_log_free(common_log * log)

    cdef void common_log_add(common_log * log, ggml.ggml_log_level level, const char * fmt, ...)

    cdef void common_log_set_file      (common_log * log, const char * file) # not thread-safe
    cdef void common_log_set_colors    (common_log * log, log_colors colors) # not thread-safe
    cdef void common_log_set_prefix    (common_log * log, bint prefix)       # whether to output prefix to each log
    cdef void common_log_set_timestamps(common_log * log, bint timestamps)   # whether to output timestamps in the prefix
    cdef void common_log_flush         (common_log * log)                    # flush all pending log messages

