cdef extern from "http_shim.h" nogil:
    ctypedef struct cy_http_server:
        pass
    ctypedef struct cy_http_res:
        pass
    ctypedef struct cy_http_sink:
        pass

    ctypedef struct cy_http_req:
        const char *method
        size_t method_len
        const char *path
        size_t path_len
        const char *auth
        size_t auth_len
        const char *body
        size_t body_len
        size_t content_length

    ctypedef int (*cy_http_handler)(void *userdata, const cy_http_req *req, cy_http_res *res) noexcept
    ctypedef int (*cy_http_stream_fn)(void *stream, cy_http_sink *sink) noexcept
    ctypedef void (*cy_http_release_fn)(void *stream) noexcept

    cy_http_server *cy_http_server_new(void *userdata, cy_http_handler precheck, cy_http_handler handle,
                                       cy_http_stream_fn next, cy_http_release_fn release, size_t max_body)
    void cy_http_server_free(cy_http_server *srv)
    int cy_http_server_bind(cy_http_server *srv, const char *host, int port, int ipv6)
    int cy_http_server_listen(cy_http_server *srv)
    void cy_http_server_stop(cy_http_server *srv)

    void cy_http_res_set(cy_http_res *res, int status, const char *content_type, const char *body, size_t len,
                         int close_connection)
    void cy_http_res_stream(cy_http_res *res, int status, const char *content_type, void *stream)
    int cy_http_sink_write(cy_http_sink *sink, const char *data, size_t len)
