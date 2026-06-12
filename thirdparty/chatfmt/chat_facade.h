// chat_facade.h - C entry point over llama.cpp's bundled Jinja engine
// (common/jinja). Lets Swift code render arbitrary chat templates
// straight from a GGUF without the hardcoded-fingerprint limitation
// of llama_chat_apply_template (see llama.h:1168 in the framework
// headers). Built into libchatfmt.dylib by scripts/make_xcframework.py
// and shipped inside LlamaCpp.framework so `import LlamaCpp` exposes it.
//
// All returned C strings are heap-allocated and must be freed with
// chatfmt_free(). NULL return signals failure; pass a non-NULL
// error_out to receive a human-readable diagnostic that the caller
// also frees with chatfmt_free().

#ifndef CYLLAMA_CHAT_FACADE_H
#define CYLLAMA_CHAT_FACADE_H

#ifdef __cplusplus
extern "C" {
#endif

// Render `template_src` (raw jinja from the GGUF's
// tokenizer.chat_template field) against `messages_json` (a JSON array
// of {"role": ..., "content": ...} objects). `bos_token` / `eos_token`
// may be NULL; they are passed as the jinja `bos_token` /
// `eos_token` variables that most templates reference. When
// `add_generation_prompt` is non-zero, the jinja
// `add_generation_prompt` variable is set true so the template emits
// its trailing assistant tag.
//
// Returns a malloc'd UTF-8 string on success (caller frees with
// chatfmt_free), or NULL on failure. When *error_out_opt is non-NULL,
// it receives a malloc'd error message; otherwise diagnostics are
// dropped.
char *chatfmt_apply(const char *template_src,
                    const char *messages_json,
                    const char *bos_token,
                    const char *eos_token,
                    int add_generation_prompt,
                    char **error_out_opt);

// free() a string returned by chatfmt_apply (result or error).
void chatfmt_free(void *p);

#ifdef __cplusplus
}
#endif

#endif  // CYLLAMA_CHAT_FACADE_H
