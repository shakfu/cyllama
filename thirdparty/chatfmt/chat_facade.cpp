// chat_facade.cpp - thin C bridge over llama.cpp's common/jinja engine.
// See chat_facade.h for the contract.

#include "chat_facade.h"

#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/value.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

namespace {

char *dup_cstr(const std::string &s) {
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

void set_error(char **error_out_opt, const std::string &msg) {
    if (error_out_opt) {
        *error_out_opt = dup_cstr(msg);
    }
}

}  // namespace

extern "C" char *chatfmt_apply(const char *template_src,
                               const char *messages_json,
                               const char *bos_token,
                               const char *eos_token,
                               int add_generation_prompt,
                               char **error_out_opt) {
    if (error_out_opt) *error_out_opt = nullptr;
    if (!template_src || !messages_json) {
        set_error(error_out_opt, "chatfmt_apply: template_src and messages_json are required");
        return nullptr;
    }

    try {
        nlohmann::ordered_json messages = nlohmann::ordered_json::parse(messages_json);
        nlohmann::ordered_json input = {
            {"messages", messages},
            {"bos_token", bos_token ? std::string(bos_token) : std::string()},
            {"eos_token", eos_token ? std::string(eos_token) : std::string()},
        };
        if (add_generation_prompt) {
            input["add_generation_prompt"] = true;
        }

        jinja::lexer lexer;
        auto lex_res = lexer.tokenize(std::string(template_src));
        jinja::program prog = jinja::parse_from_tokens(lex_res);

        jinja::context ctx(lex_res.source);
        jinja::global_from_json(ctx, input, /*mark_input=*/false);

        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        std::string rendered = parts->as_string().str();
        return dup_cstr(rendered);
    } catch (const std::exception &e) {
        set_error(error_out_opt, std::string("chatfmt_apply: ") + e.what());
        return nullptr;
    } catch (...) {
        set_error(error_out_opt, "chatfmt_apply: unknown exception");
        return nullptr;
    }
}

extern "C" void chatfmt_free(void *p) {
    std::free(p);
}
