// Stub definitions for common.h/log.h symbols.
// These are normally provided by libcommon.a, but when using dynamic linking
// against pre-built releases, libcommon is not linked.

#include <cstdarg>
#include <cstdio>

// Build info (used by static build_info string in common.h)
int LLAMA_BUILD_NUMBER = 0;
const char * LLAMA_COMMIT = "unknown";
const char * LLAMA_COMPILER = "unknown";
const char * LLAMA_BUILD_TARGET = "unknown";

// Logging stubs (used by LOG_* macros in common.h / log.h)
int common_log_verbosity_thold = 0;

struct common_log {};
static common_log _stub_log;

common_log * common_log_main() {
    return &_stub_log;
}

common_log * common_log_init() {
    return &_stub_log;
}

void common_log_free(common_log *) {}
void common_log_pause(common_log *) {}
void common_log_resume(common_log *) {}

// Forward declare ggml_log_level to match the signature in log.h
enum ggml_log_level : int;

void common_log_add(common_log *, enum ggml_log_level level, const char * fmt, ...) {
    (void)level;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

void common_log_set_verbosity_thold(int verbosity) {
    common_log_verbosity_thold = verbosity;
}

void common_log_set_file(common_log *, const char *) {}
void common_log_set_colors(common_log *, int) {}
void common_log_set_prefix(common_log *, bool) {}
void common_log_set_timestamps(common_log *, bool) {}
void common_log_flush(common_log *) {}

// String utility stubs (used by json-schema-to-grammar.cpp)
#include <string>
#include <vector>
#include <sstream>

std::string string_join(const std::vector<std::string> & values, const std::string & separator) {
    std::string result;
    for (size_t i = 0; i < values.size(); i++) {
        if (i > 0) result += separator;
        result += values[i];
    }
    return result;
}

std::string string_repeat(const std::string & str, size_t n) {
    std::string result;
    result.reserve(str.size() * n);
    for (size_t i = 0; i < n; i++) {
        result += str;
    }
    return result;
}

std::vector<std::string> string_split(const std::string & str, const std::string & delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    size_t end = str.find(delimiter);
    while (end != std::string::npos) {
        result.push_back(str.substr(start, end - start));
        start = end + delimiter.size();
        end = str.find(delimiter, start);
    }
    result.push_back(str.substr(start));
    return result;
}
