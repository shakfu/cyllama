/* C API over cpp-httplib for embedded.pyx. Callbacks run on httplib worker threads. */
#ifndef CYLLAMA_HTTP_SHIM_H
#define CYLLAMA_HTTP_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cy_http_server cy_http_server;
typedef struct cy_http_res cy_http_res;
typedef struct cy_http_sink cy_http_sink;

typedef struct {
    const char *method;
    size_t method_len;
    const char *path;
    size_t path_len;
    const char *auth; /* NULL when the Authorization header is absent */
    size_t auth_len;
    const char *body; /* empty during the precheck, which runs before the body is read */
    size_t body_len;
    size_t content_length; /* the Content-Length header, 0 if absent */
} cy_http_req;

/* precheck: return nonzero if it answered the request, which is then not routed. */
typedef int (*cy_http_handler)(void *userdata, const cy_http_req *req, cy_http_res *res);
/* Produce the next part of a stream: 1 = more to come, 0 = finished, -1 = abort. */
typedef int (*cy_http_stream_fn)(void *stream, cy_http_sink *sink);
/* Called exactly once per stream, whether it finished, aborted, or never started. */
typedef void (*cy_http_release_fn)(void *stream);

cy_http_server *cy_http_server_new(void *userdata, cy_http_handler precheck, cy_http_handler handle,
                                   cy_http_stream_fn next, cy_http_release_fn release, size_t max_body);
void cy_http_server_free(cy_http_server *srv);
/* Returns 0 on success. ipv6 selects the address family; host is never widened. */
int cy_http_server_bind(cy_http_server *srv, const char *host, int port, int ipv6);
/* Serve until cy_http_server_stop. Blocks; call without the GIL. */
int cy_http_server_listen(cy_http_server *srv);
void cy_http_server_stop(cy_http_server *srv);

void cy_http_res_set(cy_http_res *res, int status, const char *content_type, const char *body, size_t len,
                     int close_connection);
void cy_http_res_stream(cy_http_res *res, int status, const char *content_type, void *stream);
/* Returns 0 once the client has gone. */
int cy_http_sink_write(cy_http_sink *sink, const char *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif
