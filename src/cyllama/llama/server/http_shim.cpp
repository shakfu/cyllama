#include "http_shim.h"

#include <memory>
#include <string>

#include "httplib.h"

struct cy_http_server {
    httplib::Server svr;
};

struct cy_http_res {
    httplib::Response *res;
    cy_http_release_fn release;
    cy_http_stream_fn next;
};

struct cy_http_sink {
    httplib::DataSink *sink;
};

static cy_http_req make_req(const httplib::Request &req, const std::string &auth, bool has_auth) {
    cy_http_req r;
    r.method = req.method.data();
    r.method_len = req.method.size();
    r.path = req.path.data();
    r.path_len = req.path.size();
    r.auth = has_auth ? auth.data() : nullptr;
    r.auth_len = auth.size();
    r.body = req.body.data();
    r.body_len = req.body.size();
    r.content_length = req.get_header_value_u64("Content-Length");
    return r;
}

extern "C" {

cy_http_server *cy_http_server_new(void *userdata, cy_http_handler precheck, cy_http_handler handle,
                                   cy_http_stream_fn next, cy_http_release_fn release, size_t max_body) {
    auto *s = new cy_http_server();
    auto &svr = s->svr;

    // httplib's default sets SO_REUSEPORT, which lets a second process bind the same port.
    svr.set_socket_options([](socket_t sock) {
#ifdef _WIN32
        httplib::set_socket_opt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
        httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
    });
    // Backstop for bodies without Content-Length; the precheck handles the rest before reading.
    svr.set_payload_max_length(max_body);

    svr.set_pre_routing_handler([=](const httplib::Request &req, httplib::Response &res) {
        bool has_auth = req.has_header("Authorization");
        std::string auth = req.get_header_value("Authorization");
        cy_http_req r = make_req(req, auth, has_auth);
        cy_http_res out{&res, release, next};
        return precheck(userdata, &r, &out) ? httplib::Server::HandlerResponse::Handled
                                            : httplib::Server::HandlerResponse::Unhandled;
    });

    auto route = [=](const httplib::Request &req, httplib::Response &res) {
        bool has_auth = req.has_header("Authorization");
        std::string auth = req.get_header_value("Authorization");
        cy_http_req r = make_req(req, auth, has_auth);
        cy_http_res out{&res, release, next};
        handle(userdata, &r, &out);
    };
    svr.Get(".*", route);
    svr.Post(".*", route);
    svr.Put(".*", route);
    svr.Patch(".*", route);
    svr.Delete(".*", route);
    svr.Options(".*", route);
    return s;
}

void cy_http_server_free(cy_http_server *srv) { delete srv; }

int cy_http_server_bind(cy_http_server *srv, const char *host, int port, int ipv6) {
    srv->svr.set_address_family(ipv6 ? AF_INET6 : AF_INET);
    return srv->svr.bind_to_port(host, port) ? 0 : -1;
}

int cy_http_server_listen(cy_http_server *srv) { return srv->svr.listen_after_bind() ? 0 : -1; }

void cy_http_server_stop(cy_http_server *srv) { srv->svr.stop(); }

void cy_http_res_set(cy_http_res *res, int status, const char *content_type, const char *body, size_t len,
                     int close_connection) {
    res->res->status = status;
    res->res->set_content(body, len, content_type);
    if (close_connection) res->res->set_header("Connection", "close");
}

void cy_http_res_stream(cy_http_res *res, int status, const char *content_type, void *stream) {
    // httplib may copy the provider; the shared_ptr releases the stream once, after the last copy.
    auto release = res->release;
    std::shared_ptr<void> st(stream, [release](void *p) { release(p); });
    auto next = res->next;
    res->res->status = status;
    res->res->set_chunked_content_provider(content_type, [st, next](size_t, httplib::DataSink &sink) {
        cy_http_sink s{&sink};
        int r = next(st.get(), &s);
        if (r == 0) sink.done();
        return r >= 0;
    });
}

int cy_http_sink_write(cy_http_sink *sink, const char *data, size_t len) {
    return sink->sink->write(data, len) ? 1 : 0;
}

}  // extern "C"
