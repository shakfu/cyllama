/*
 * C wrapper for mongoose.c to handle compilation separately from C++ code
 * This allows us to compile mongoose.c with C flags and the rest with C++
 */

#include <stdarg.h>
#include <stdio.h>
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

void cyllama_mg_http_reply(struct mg_connection *c, int status_code, const char *headers,
                          const char *body_fmt, ...) {
    va_list ap;
    va_start(ap, body_fmt);

    /* Create a formatted string from the body_fmt and args */
    char body[4096];
    vsnprintf(body, sizeof(body), body_fmt, ap);
    va_end(ap);

    mg_http_reply(c, status_code, headers, "%s", body);
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