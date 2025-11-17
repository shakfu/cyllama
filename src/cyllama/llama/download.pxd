# cython: language_level=3
# download.pxd - Cython declarations for download.h

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

# Forward declarations
cdef extern from "common.h":
    cdef struct common_params_model:
        string path         # model local path
        string url          # model url to download
        string hf_repo      # HF repo
        string hf_file      # HF file
        string docker_repo  # Docker repo

cdef extern from "download.h":
    cdef struct common_cached_model_info:
        string manifest_path
        string user
        string model
        string tag
        size_t size  # GGUF size in bytes
        string to_string()

    cdef struct common_hf_file_res:
        string repo       # repo name with ":tag" removed
        string ggufFile
        string mmprojFile

    # Get HF file from HF repo with tag (like ollama)
    # Examples:
    # - bartowski/Llama-3.2-3B-Instruct-GGUF:q4
    # - bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
    # Tag is optional, default to "latest"
    common_hf_file_res common_get_hf_file(
        const string & hf_repo_with_tag,
        const string & bearer_token,
        bool offline
    ) except +

    # Download model, returns true if download succeeded
    bool common_download_model(
        const common_params_model & model,
        const string & bearer_token,
        bool offline
    ) except +

    # List cached models
    vector[common_cached_model_info] common_list_cached_models() except +

    # Resolve and download model from Docker registry
    # Return local path to downloaded model file
    string common_docker_resolve_model(const string & docker) except +
