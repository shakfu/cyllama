# distutils: language=c++

from libc.stdint cimport int32_t, int8_t, int64_t, uint32_t, uint64_t, uint8_t
from libc.stdio cimport FILE
from libcpp.string cimport string as std_string
from libcpp.vector cimport vector as std_vector
from libcpp.set cimport set as std_set
from libcpp.memory cimport unique_ptr as std_unique_ptr
from libcpp.set cimport set as std_set
from libcpp.functional cimport function as std_function
from libcpp.map cimport map as std_map

cimport common
cimport llama

#------------------------------------------------------------------------------
# chat.h

cdef extern from "chat.h":

    # Forward declaration
    ctypedef struct common_chat_templates:
        pass

    # Tool call structure
    ctypedef struct common_chat_tool_call:
        std_string name
        std_string arguments
        std_string id

    # Message content part structure
    ctypedef struct common_chat_msg_content_part:
        std_string type
        std_string text

    # Main chat message structure
    ctypedef struct common_chat_msg:
        std_string role
        std_string content
        std_vector[common_chat_msg_content_part] content_parts
        std_vector[common_chat_tool_call] tool_calls
        std_string reasoning_content
        std_string tool_name
        std_string tool_call_id

    # Message diff structure
    ctypedef struct common_chat_msg_diff:
        std_string reasoning_content_delta
        std_string content_delta
        size_t tool_call_index
        common_chat_tool_call tool_call_delta

    # Tool structure
    ctypedef struct common_chat_tool:
        std_string name
        std_string description
        std_string parameters

    # Tool choice enum
    ctypedef enum common_chat_tool_choice:
        COMMON_CHAT_TOOL_CHOICE_AUTO
        COMMON_CHAT_TOOL_CHOICE_REQUIRED
        COMMON_CHAT_TOOL_CHOICE_NONE

    # Chat format enum
    ctypedef enum common_chat_format:
        COMMON_CHAT_FORMAT_CONTENT_ONLY
        COMMON_CHAT_FORMAT_GENERIC
        COMMON_CHAT_FORMAT_MISTRAL_NEMO
        COMMON_CHAT_FORMAT_LLAMA_3_X
        COMMON_CHAT_FORMAT_LLAMA_3_X_WITH_BUILTIN_TOOLS
        COMMON_CHAT_FORMAT_DEEPSEEK_R1
        COMMON_CHAT_FORMAT_FIREFUNCTION_V2
        COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2
        COMMON_CHAT_FORMAT_FUNCTIONARY_V3_1_LLAMA_3_1
        COMMON_CHAT_FORMAT_DEEPSEEK_V3_1
        COMMON_CHAT_FORMAT_HERMES_2_PRO
        COMMON_CHAT_FORMAT_COMMAND_R7B
        COMMON_CHAT_FORMAT_GRANITE,
        COMMON_CHAT_FORMAT_GPT_OSS
        COMMON_CHAT_FORMAT_SEED_OSS
        COMMON_CHAT_FORMAT_NEMOTRON_V2
        COMMON_CHAT_FORMAT_COUNT

    # Chat templates inputs structure
    ctypedef struct common_chat_templates_inputs:
        std_vector[common_chat_msg] messages
        std_string grammar
        std_string json_schema
        bint add_generation_prompt
        bint use_jinja
        std_vector[common_chat_tool] tools
        common_chat_tool_choice tool_choice
        bint parallel_tool_calls
        common.common_reasoning_format reasoning_format
        bint enable_thinking
        # std::chrono::system_clock::time_point now - not directly supported in Cython
        std_map[std_string, std_string] chat_template_kwargs
        bint add_bos
        bint add_eos


    # Chat parameters structure
    ctypedef struct common_chat_params:
        common_chat_format format
        std_string prompt
        std_string grammar
        bint grammar_lazy
        bint thinking_forced_open
        std_vector[common.common_grammar_trigger] grammar_triggers
        std_vector[std_string] preserved_tokens
        std_vector[std_string] additional_stops

    # Chat syntax structure
    ctypedef struct common_chat_syntax:
        common_chat_format format
        common.common_reasoning_format reasoning_format
        bint reasoning_in_content
        bint thinking_forced_open
        bint parse_tool_calls

    # Template deleter structure
    ctypedef struct common_chat_templates_deleter:
        pass

    # Template pointer type
    ctypedef std_unique_ptr[common_chat_templates, common_chat_templates_deleter] common_chat_templates_ptr

    # Function declarations
    cdef bint common_chat_verify_template(const std_string & tmpl, bint use_jinja)
    
    cdef void common_chat_templates_free(common_chat_templates * tmpls)
    
    cdef common_chat_templates_ptr common_chat_templates_init(
        const llama.llama_model * model,
        const std_string & chat_template_override,
        const std_string & bos_token_override,
        const std_string & eos_token_override)
    
    cdef bint common_chat_templates_was_explicit(const common_chat_templates * tmpls)
    
    cdef const char * common_chat_templates_source(const common_chat_templates * tmpls, const char * variant)
    
    cdef common_chat_params common_chat_templates_apply(
        const common_chat_templates * tmpls,
        const common_chat_templates_inputs & inputs)
    
    cdef std_string common_chat_format_single(
        const common_chat_templates * tmpls,
        const std_vector[common_chat_msg] & past_msg,
        const common_chat_msg & new_msg,
        bint add_ass,
        bint use_jinja)
    
    cdef std_string common_chat_format_example(
        const common_chat_templates * tmpls,
        bint use_jinja)
    
    cdef const char* common_chat_format_name(common_chat_format format)
    
    cdef const char* common_reasoning_format_name(common.common_reasoning_format format)
    
    cdef common_chat_msg common_chat_parse(const std_string & input, bint is_partial, const common_chat_syntax & syntax)
    
    cdef common_chat_tool_choice common_chat_tool_choice_parse_oaicompat(const std_string & tool_choice)

    cdef bint common_chat_templates_support_enable_thinking(const common_chat_templates * chat_templates)

    # Legacy structures for backward compatibility
    ctypedef struct chat_completion_chunk:
        std_string role
        std_string content
        std_string finish_reason

    ctypedef struct chat_completion_message:
        std_string role
        std_string content

    ctypedef struct chat_completion_usage:
        int64_t prompt_tokens
        int64_t completion_tokens
        int64_t total_tokens

    ctypedef struct chat_completion:
        std_vector[chat_completion_message] messages
        std_string model
        float temperature
    
    