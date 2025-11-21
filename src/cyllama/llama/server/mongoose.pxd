# Cython declarations for Mongoose embedded web server
# Based on mongoose.h

from libc.stdint cimport uint64_t, uint16_t, uint8_t
from libc.stdio cimport FILE

cdef extern from "mongoose_wrapper.c":
    pass

cdef extern from "mongoose.h":
    # Basic types
    ctypedef struct mg_str:
        char *buf
        size_t len

    # Event types
    cdef enum:
        MG_EV_ERROR      # Error                        char *error_message
        MG_EV_OPEN       # Connection created           NULL
        MG_EV_POLL       # mg_mgr_poll iteration        uint64_t *milliseconds
        MG_EV_RESOLVE    # Host name is resolved        NULL
        MG_EV_CONNECT    # Connection established       NULL
        MG_EV_ACCEPT     # Connection accepted          NULL
        MG_EV_TLS_HS     # TLS handshake succeeded      NULL
        MG_EV_READ       # Data received from socket    long *bytes_read
        MG_EV_WRITE      # Data written to socket       long *bytes_written
        MG_EV_CLOSE      # Connection closed            NULL
        MG_EV_HTTP_HDRS  # HTTP headers                 mg_http_message *
        MG_EV_HTTP_MSG   # Full HTTP request/response   mg_http_message *
        MG_EV_WS_OPEN    # Websocket handshake done     mg_http_message *
        MG_EV_WS_MSG     # Websocket msg, text or bin   mg_ws_message *
        MG_EV_WS_CTL     # Websocket control msg        mg_ws_message *

    # HTTP structures
    ctypedef struct mg_http_header:
        mg_str name
        mg_str value

    ctypedef struct mg_http_message:
        mg_str method
        mg_str uri
        mg_str query
        mg_str proto
        mg_http_header headers[30]  # MG_MAX_HTTP_HEADERS is 30
        mg_str body

    # Connection and manager structures
    ctypedef struct mg_connection:
        mg_connection *next
        void *mgr
        unsigned long id
        void *fd
        mg_str recv
        mg_str send
        mg_str rtls
        mg_str peer
        mg_str data  # Application-specific data
        void *pfn_data
        void *fn_data
        char is_listening
        char is_client
        char is_accepted
        char is_resolving
        char is_connecting
        char is_tls
        char is_tls_hs
        char is_udp
        char is_websocket
        char is_hexdumping
        char is_draining
        char is_closing
        char is_full
        char is_resp
        char is_readable
        char is_writable

    ctypedef void (*mg_event_handler_t)(mg_connection *c, int ev, void *ev_data, void *fn_data)

    ctypedef struct mg_mgr:
        mg_connection *conns
        mg_str extraconninfo
        int use_dns_cache
        int epoll_fd
        void *userdata

    # Core API functions (using wrapper) - optimized with nogil
    void cyllama_mg_mgr_init(mg_mgr *mgr) nogil
    void cyllama_mg_mgr_free(mg_mgr *mgr) nogil
    void cyllama_mg_mgr_poll(mg_mgr *mgr, int timeout_ms) nogil

    mg_connection *cyllama_mg_http_listen(mg_mgr *mgr, const char *url,
                                          mg_event_handler_t fn, void *fn_data) nogil

    # HTTP response functions (using wrapper) - optimized with nogil
    void cyllama_mg_http_reply(mg_connection *c, int status_code, const char *headers,
                               const char *body_fmt, ...) nogil

    # Helper functions for HTTP (using wrapper) - optimized with nogil
    mg_str *cyllama_mg_http_get_header(mg_http_message *hm, const char *name) nogil
    int cyllama_mg_http_get_var(const mg_str *buf, const char *name, char *dst, size_t dst_len) nogil

    # String utilities (using wrapper) - optimized with nogil
    mg_str cyllama_mg_str_n(const char *s, size_t n) nogil
    mg_str cyllama_mg_str(const char *s) nogil

    # Print utilities - optimized with nogil
    size_t mg_printf(mg_connection *c, const char *fmt, ...) nogil
    bint mg_send(mg_connection *c, const void *buf, size_t len) nogil