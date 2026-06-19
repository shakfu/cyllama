#include "json-schema-to-grammar.h"
#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <signal.h>

#include <atomic>
#include <clocale>
#include <exception>
#include <future>
#include <thread>  // for std::thread::hardware_concurrency

#if defined(_WIN32)
#    include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag         is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    if (shutdown_handler) {
        shutdown_handler(signal);
    }
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type  error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error   = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error   = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error   = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res    = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status     = json_value(error_data, "code", 500);
            res->data       = safe_json_to_str({
                { "error", error_data }
            });
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

static void init(common_params &   params,
                 server_context &  ctx_server,
                 std::string &     listening_address,
                 std::promise<int> out) {
    // Do not call std::setlocale() in this Python extension module: it mutates
    // the process-wide locale used by the host interpreter and other libraries.
    // std::setlocale(LC_NUMERIC, "C");

    common_log_set_verbosity_thold(params.verbosity);

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    SRV_INF("build_info: %s\n", llama_build_info());

    const bool is_router_server = false;
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        out.set_value(1);
        return;
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);
    server_tools  tools;

    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, 0, nullptr);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            out.set_value(1);
            return;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props  = models_routes->get_router_props;
        routes.get_models = models_routes->get_router_models;

        ctx_http.post("/models", ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load", ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload", ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get("/models/sse", ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del("/models", ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get("/health", ex_wrapper(routes.get_health));     // public endpoint (no API key check)
    ctx_http.get("/v1/health", ex_wrapper(routes.get_health));  // public endpoint (no API key check)
    ctx_http.get("/metrics", ex_wrapper(routes.get_metrics));
    ctx_http.get("/props", ex_wrapper(routes.get_props));
    ctx_http.post("/props", ex_wrapper(routes.post_props));
    ctx_http.get("/models", ex_wrapper(routes.get_models));             // public endpoint (no API key check)
    ctx_http.get("/v1/models", ex_wrapper(routes.get_models));          // public endpoint (no API key check)
    ctx_http.post("/completion", ex_wrapper(routes.post_completions));  // legacy
    ctx_http.post("/completions", ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions", ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses", ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses", ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions", ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions", ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages", ex_wrapper(routes.post_anthropic_messages));  // anthropic messages API
    ctx_http.post("/infill", ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding", ex_wrapper(routes.post_embeddings));            // legacy
    ctx_http.post("/embeddings", ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings", ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank", ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking", ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank", ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking", ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize", ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize", ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template", ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens", ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens", ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",
                  ex_wrapper(routes.post_anthropic_count_tokens));  // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get("/lora-adapters", ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters", ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get("/slots", ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot", ex_wrapper(routes.post_slots));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    // Supports both new ui_mcp_proxy and deprecated webui_mcp_proxy fields
    if (params.ui_mcp_proxy || params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get("/cors-proxy", ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy", ex_wrapper(proxy_handler_post));
    }
    // EXPERIMENTAL built-in tools
    if (!params.server_tools.empty()) {
        try {
            tools.setup(params.server_tools);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            out.set_value(1);
            return;
        }
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get("/tools", ex_wrapper(tools.handle_get));
        ctx_http.post("/tools", ex_wrapper(tools.handle_post));
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting router server, no model will be loaded in this process\n");

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->stopping.store(true);  // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            out.set_value(1);
            return;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            out.set_value(1);
            return;
        }

        // load the model
        SRV_INF("%s", "loading model\n");

        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed(
                [&](bool sleeping) { server_models::notify_router_sleeping_state(sleeping); });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            out.set_value(1);
            return;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // TODO: refactor in common/console
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        SRV_INF("router server is listening on %s\n", ctx_http.listening_address.c_str());
        SRV_WRN("%s", "NOTE: router mode is experimental\n");
        SRV_WRN("%s", "      it is not recommended to use this mode in untrusted environments\n");
        if (!params.models_preset_hf.empty()) {
            SRV_WRN("NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();  // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            json model_info = routes.get_model_info();
            monitor_thread  = server_models::setup_child_server(shutdown_handler, model_info);
        }

        // write the listening_address
        listening_address = ctx_http.listening_address;

        out.set_value(0);

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        // crash during common_memory_breakdown_print if the model is rerank.
        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr && params.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    // init() runs on the Python Server thread; clear the process-wide handler
    // before its lambdas can outlive ctx_http / ctx_server stack references.
    shutdown_handler = nullptr;
}

static void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) text;
    (void) user_data;
    // if (level == GGML_LOG_LEVEL_INFO || level == GGML_LOG_LEVEL_ERROR) {
    //   fputs(text, stderr);
    //   fflush(stderr);
    // }
}

std::function<bool()> not_stop = [] {
    return false;
};

static std::vector<std::string> parse_oai_sse(const std::string & sse) {
    std::vector<std::string> out;

    std::size_t start = 0;
    while (start < sse.size()) {
        std::size_t end = sse.find('\n', start);
        if (end == std::string::npos) {
            break;
        }

        // Empty line = event separator, skip
        if (end > start) {
            // Guaranteed format: "data: <json>"
            out.emplace_back(sse.substr(start + 6, end - start - 6));
        }

        start = end + 1;
    }

    return out;
}

static void process_handler_response(server_http_res_ptr &               response,
                                     std::function<bool(std::string &&)> res_err,
                                     std::function<bool(std::string &&)> res_ok) {
    static const std::string sse_prefix("data: ");
    auto                     res = response->status == 200 ? res_ok : res_err;
    if (response->is_stream()) {
        std::string chunk;

        while (true) {
            const bool has_next = response->next(chunk);
            if (!chunk.empty() && chunk.size() >= sse_prefix.size()) {
                if (!has_next && chunk == "data: [DONE]\n\n") {
                    return;
                }
                auto parsed = parse_oai_sse(chunk);
                for (auto && json_str : parsed) {
                    if (res(std::move(json_str))) {
                        return;
                    }
                }
            }
            if (!has_next) {
                return;
            }
        }
    } else {
        res(std::move(response->data));
    }
}

#include "server.h"

namespace xllamacpp {

std::string get_system_info() {
    return llama_print_system_info();
}

std::vector<ggml_backend_dev_props> get_device_info() {
    ggml_log_set(ggml_log_callback_default, nullptr);

    const size_t dev_count = ggml_backend_dev_count();

    std::vector<ggml_backend_dev_props> result;
    std::vector<ggml_backend_dev_t>     devs;
    std::vector<ggml_backend_t>         backends;

    for (size_t i = 0; i < dev_count; ++i) {
        devs.push_back(ggml_backend_dev_get(i));

        ggml_backend_t backend = ggml_backend_dev_init(devs[i], NULL);
        GGML_ASSERT(backend != NULL);

        auto * reg = ggml_backend_dev_backend_reg(devs[i]);
        auto   ggml_backend_set_n_threads_fn =
            (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (ggml_backend_set_n_threads_fn) {
            ggml_backend_set_n_threads_fn(backend, std::thread::hardware_concurrency() / 2);
        }

        backends.push_back(backend);
    }

    for (size_t i = 0; i < dev_count; ++i) {
        // Put the backend to be tested in front so that it's prioritized:
        std::vector<ggml_backend_t> backends_modded = { backends[i] };
        backends_modded.insert(backends_modded.end(), backends.begin(), backends.end());

        ggml_backend_dev_props prop;
        ggml_backend_dev_get_props(devs[i], &prop);
        // Avoid crash when converting the prop struct to Python dict by Cython.
        if (prop.device_id == nullptr) {
            prop.device_id = "";
        }

        result.push_back(prop);
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_free(backend);
    }

    return result;
}

Server::Server(const common_params & params) : _params(params), _ctx_server(new server_context()) {
    if (_params.model.path.empty()) {
        throw std::invalid_argument("params.model.path is required: xllamacpp disabled the router server feature");
    }

    std::promise<int> out;
    std::future<int>  fut = out.get_future();
    _loop_thread =
        std::thread(init, std::ref(_params), std::ref(*_ctx_server), std::ref(_listening_address), std::move(out));
    if (fut.get() != 0) {
        if (_loop_thread.joinable()) {
            _loop_thread.join();
        }
        throw std::runtime_error("Failed to init server, please check the input params.");
    }

    _routes = std::make_shared<server_routes>(_params, *_ctx_server);
    _routes->update_meta(*_ctx_server);
}

Server::~Server() {
    _ctx_server->terminate();
    LOG_INF("%s: waiting for main loop exit\n", __func__);
    if (_loop_thread.joinable()) {
        _loop_thread.join();
    }
    LOG_INF("%s: main loop exited\n", __func__);
}

std::string Server::listening_address() const {
    return _listening_address;
}

std::string Server::handle_metrics() {
    server_http_req req{ {}, {}, "", "", "", {}, not_stop };
    auto            res = _routes->get_metrics(req);
    return res->data;
}

std::string Server::handle_embeddings(const std::string & input_json_str) {
    server_http_req req{ {}, {}, "", "", input_json_str, {}, not_stop };
    auto            res = _routes->post_embeddings_oai(req);
    return res->data;
}

std::string Server::handle_rerank(const std::string & input_json_str) {
    server_http_req req{ {}, {}, "", "", input_json_str, {}, not_stop };
    auto            res = _routes->post_rerank(req);
    return res->data;
}

void Server::handle_completions(const std::string & prompt_json_str,
                                Callback            res_err,
                                void *              py_cb_err,
                                Callback            res_ok,
                                void *              py_cb_ok) {
    server_http_req req{ {}, {}, "", "", prompt_json_str, {}, not_stop };
    auto            res = _routes->post_completions_oai(req);
    process_handler_response(
        res, [res_err, py_cb_err](std::string && err) { return res_err(std::move(err), py_cb_err); },
        [res_ok, py_cb_ok](std::string && ok) { return res_ok(std::move(ok), py_cb_ok); });
}

void Server::handle_chat_completions(const std::string & prompt_json_str,
                                     Callback            res_err,
                                     void *              py_cb_err,
                                     Callback            res_ok,
                                     void *              py_cb_ok) {
    server_http_req req{ {}, {}, "", "", prompt_json_str, {}, not_stop };
    auto            res = _routes->post_chat_completions(req);
    process_handler_response(
        res, [res_err, py_cb_err](std::string && err) { return res_err(std::move(err), py_cb_err); },
        [res_ok, py_cb_ok](std::string && ok) { return res_ok(std::move(ok), py_cb_ok); });
}

std::string json_schema_to_grammar_str(const std::string & schema_json_str) {
    try {
        auto schema = json::parse(schema_json_str);
        return json_schema_to_grammar(schema);
    } catch (const std::exception & e) {
        throw std::runtime_error(std::string("json_schema_to_grammar: ") + e.what());
    }
}

// Helper function to parse tensor buffer override strings
void parse_tensor_buffer_overrides(const std::string &                             value,
                                   std::vector<llama_model_tensor_buft_override> & overrides) {
    std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev  = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[ggml_backend_buft_name(buft)] = buft;
        }
    }

    for (const auto & override : string_split<std::string>(value, ',')) {
        std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            throw std::invalid_argument("invalid value");
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);

        if (buft_list.find(buffer_type) == buft_list.end()) {
            printf("Available buffer types:\n");
            for (const auto & it : buft_list) {
                printf("  %s\n", ggml_backend_buft_name(it.second));
            }
            throw std::invalid_argument("unknown buffer type");
        }
        // keep strings alive and avoid leaking memory by storing them in a static
        // vector
        static std::list<std::string> buft_overrides;
        buft_overrides.push_back(tensor_name);
        overrides.push_back({ buft_overrides.back().c_str(), buft_list.at(buffer_type) });
    }
}

// Helper function to build tensor buffer override strings
void build_tensor_buffer_overrides(const std::vector<llama_model_tensor_buft_override> & overrides,
                                   std::string &                                         value) {
    std::map<ggml_backend_buffer_type_t, std::string> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev  = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[buft] = ggml_backend_buft_name(buft);
        }
    }

    std::vector<std::string> parts;
    for (auto & override : overrides) {
        std::string ov_str = std::string(override.pattern) + "=" + buft_list[override.buft];
        parts.emplace_back(ov_str);
    }

    value = string_join(parts, ",");
}

// Helper function to parse device list
std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto                            dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

// Helper function to build device string from vector of ggml_backend_dev_t
std::string build_device_string(const std::vector<ggml_backend_dev_t> & devices) {
    if (devices.empty()) {
        return "";
    }
    if (devices.size() == 1 && devices[0] == nullptr) {
        return "";
    }
    std::vector<std::string> names;
    for (size_t i = 0; i < devices.size() - 1; ++i) {  // Skip the trailing nullptr
        if (devices[i]) {
            names.emplace_back(ggml_backend_dev_name(devices[i]));
        }
    }
    return string_join(names, ",");
}

}  // namespace xllamacpp
