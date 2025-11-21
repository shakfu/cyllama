#include "json_schema.h"
#include "json-partial.h"
#include "json-schema-to-grammar.h"
#include <cstring>
#include <cstdlib>
#include <string>

extern "C" {

char* json_schema_to_grammar_wrapper(const char* json_schema_str, bool force_gbnf) {
    if (!json_schema_str) {
        return nullptr;
    }

    try {
        // Parse JSON string using common_json_parse
        std::string input(json_schema_str);
        common_json parsed;

        if (!common_json_parse(input, "", parsed)) {
            return nullptr;
        }

        // Convert to grammar
        std::string grammar = json_schema_to_grammar(parsed.json, force_gbnf);

        // Allocate C string
        char* result = static_cast<char*>(malloc(grammar.size() + 1));
        if (!result) {
            return nullptr;
        }

        std::memcpy(result, grammar.c_str(), grammar.size() + 1);
        return result;
    } catch (...) {
        // Return nullptr on any error (parse error, etc.)
        return nullptr;
    }
}

}
