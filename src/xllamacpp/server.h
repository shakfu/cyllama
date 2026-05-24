#include "common.h"

#include <memory>
#include <string>
#include <thread>
#include <vector>

struct server_context;
struct server_routes;

namespace xllamacpp {

std::string get_system_info();

std::vector<ggml_backend_dev_props> get_device_info();

typedef bool (*Callback)(std::string &&, void * py_cb);

// Convert a JSON schema string into a llama.cpp grammar string for structured
// outputs
std::string json_schema_to_grammar_str(const std::string & schema_json_str);

class Server {
  public:
    Server(const common_params & params);
    ~Server();

    std::string listening_address() const;

    std::string handle_metrics();

    std::string handle_embeddings(const std::string & input_json_str);

    std::string handle_rerank(const std::string & input_json_str);

    void handle_completions(const std::string & prompt_json_str,
                            Callback            res_error,
                            void *              py_cb_error,
                            Callback            res_ok,
                            void *              py_cb_ok);

    void handle_chat_completions(const std::string & prompt_json_str,
                                 Callback            res_error,
                                 void *              py_cb_error,
                                 Callback            res_ok,
                                 void *              py_cb_ok);

  private:
    common_params                   _params;
    std::string                     _listening_address;
    // Incomplete type of server_context
    std::shared_ptr<server_context> _ctx_server;
    std::shared_ptr<server_routes>  _routes;
    std::thread                     _loop_thread;
};

void parse_tensor_buffer_overrides(const std::string &                             value,
                                   std::vector<llama_model_tensor_buft_override> & overrides);
void build_tensor_buffer_overrides(const std::vector<llama_model_tensor_buft_override> & overrides,
                                   std::string &                                         value);
std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value);
std::string                     build_device_string(const std::vector<ggml_backend_dev_t> & devices);
}  // namespace xllamacpp
