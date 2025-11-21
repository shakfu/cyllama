#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Convert JSON schema string to GBNF grammar string
// Returns nullptr on error
// Caller must free the returned string with free()
char* json_schema_to_grammar_wrapper(const char* json_schema_str, bool force_gbnf);

#ifdef __cplusplus
}
#endif
