/*
 * C wrapper for mongoose.c to handle compilation separately from C++ code
 * This allows us to compile mongoose.c with C flags and the rest with C++
 */

#include "mongoose.h"

/* Re-export all mongoose functions we need for Cython */

void cyllama_mg_mgr_init(struct mg_mgr *mgr) {
    mg_mgr_init(mgr);
}

void cyllama_mg_mgr_free(struct mg_mgr *mgr) {
    mg_mgr_free(mgr);
}

void cyllama_mg_mgr_poll(struct mg_mgr *mgr, int timeout_ms) {
    mg_mgr_poll(mgr, timeout_ms);
}

struct mg_connection *cyllama_mg_http_listen(struct mg_mgr *mgr, const char *url,
                                            mg_event_handler_t fn, void *fn_data) {
    return mg_http_listen(mgr, url, fn, fn_data);
}

/* Length-explicit: Mongoose's send buffer grows as needed, so no intermediate
   buffer (a fixed 4096-byte one truncated larger responses). */
void cyllama_mg_http_reply(struct mg_connection *c, int status_code, const char *headers,
                          const char *body, size_t body_len) {
    mg_http_reply(c, status_code, headers, "%.*s", (int) body_len, body);
}

struct mg_str *cyllama_mg_http_get_header(struct mg_http_message *hm, const char *name) {
    return mg_http_get_header(hm, name);
}

int cyllama_mg_http_get_var(const struct mg_str *buf, const char *name, char *dst, size_t dst_len) {
    return mg_http_get_var(buf, name, dst, dst_len);
}

/* Helper functions for string handling */
struct mg_str cyllama_mg_str(const char *s) {
    return mg_str(s);
}

struct mg_str cyllama_mg_str_n(const char *s, size_t n) {
    return mg_str_n(s, n);
}