import pytest
# pytest.skip(allow_module_level=True)

import platform

from pytest import approx

PLATFORM = platform.system()
# ARCH = platform.machine()

import cyllama.llama.llama_cpp as cy


def test_default_model_params():
    params = cy.LlamaModelParams()
    # n_gpu_layers defaults to -1 (auto-detect: use all available GPU layers)
    assert params.n_gpu_layers == -1
    assert params.split_mode == 1  # LLAMA_SPLIT_MODE_LAYER = 1
    assert params.main_gpu == 0
    assert params.vocab_only == False
    # llama.cpp v0.3.0 made AUTO (-1) the default: the loader picks the mode
    # from the device's capabilities rather than always memory-mapping.
    assert params.load_mode == cy.LLAMA_LOAD_MODE_AUTO
    assert params.load_mode_name == "auto"
    assert params.check_tensors == False
    assert params.progress_callback is None
    assert params.tensor_split == []  # Default is empty (no custom split)


def test_model_params_load_mode():
    """load_mode replaces the removed use_mmap/use_mlock/use_direct_io fields."""
    params = cy.LlamaModelParams()

    assert cy.LLAMA_LOAD_MODE_AUTO == -1
    assert cy.LLAMA_LOAD_MODE_NONE == 0
    assert cy.LLAMA_LOAD_MODE_MMAP == 1
    assert cy.LLAMA_LOAD_MODE_MLOCK == 2
    assert cy.LLAMA_LOAD_MODE_MMAP_MLOCK == 3
    assert cy.LLAMA_LOAD_MODE_DIRECT_IO == 4

    for mode, name in [
        (cy.LLAMA_LOAD_MODE_AUTO, "auto"),
        (cy.LLAMA_LOAD_MODE_NONE, "none"),
        (cy.LLAMA_LOAD_MODE_MMAP, "mmap"),
        (cy.LLAMA_LOAD_MODE_MLOCK, "mlock"),
        (cy.LLAMA_LOAD_MODE_MMAP_MLOCK, "mmap+mlock"),
        (cy.LLAMA_LOAD_MODE_DIRECT_IO, "dio"),
    ]:
        params.load_mode = mode
        assert params.load_mode == mode
        assert params.load_mode_name == name


def test_model_params_removed_load_flags():
    """The pre-b10107 booleans are gone; load_mode is the only way to set this."""
    params = cy.LlamaModelParams()
    for attr in ("use_mmap", "use_mlock", "use_direct_io"):
        assert not hasattr(params, attr)
        with pytest.raises(AttributeError):
            setattr(params, attr, True)


def test_model_params_tensor_split():
    """Test that tensor_split can be set and retrieved."""
    params = cy.LlamaModelParams()

    # Initially empty
    assert params.tensor_split == []

    # Set tensor_split
    params.tensor_split = [0.5, 0.5]
    result = params.tensor_split
    assert len(result) == cy.llama_max_devices()
    assert result[0] == approx(0.5, rel=1e-5)
    assert result[1] == approx(0.5, rel=1e-5)
    # Remaining should be zero-filled
    assert all(v == 0.0 for v in result[2:])

    # Set different values
    params.tensor_split = [1.0, 2.0, 1.0]
    result = params.tensor_split
    assert result[0] == approx(1.0, rel=1e-5)
    assert result[1] == approx(2.0, rel=1e-5)
    assert result[2] == approx(1.0, rel=1e-5)

    # Clear with None
    params.tensor_split = None
    assert params.tensor_split == []

    # Set again after clearing
    params.tensor_split = [0.3, 0.7]
    result = params.tensor_split
    assert result[0] == approx(0.3, rel=1e-5)
    assert result[1] == approx(0.7, rel=1e-5)

    # Clear with empty list
    params.tensor_split = []
    assert params.tensor_split == []


def test_model_params_tensor_split_validation():
    """Test tensor_split validation."""
    params = cy.LlamaModelParams()

    # Should raise if too many elements
    max_devices = cy.llama_max_devices()
    with pytest.raises(ValueError):
        params.tensor_split = [0.5] * (max_devices + 1)


def test_model_params_progress_callback():
    """Test that progress callback can be set and retrieved."""
    params = cy.LlamaModelParams()

    # Initially None
    assert params.progress_callback is None

    # Set a callback
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        return True  # continue loading

    params.progress_callback = on_progress
    assert params.progress_callback is on_progress

    # Set to None to disable
    params.progress_callback = None
    assert params.progress_callback is None


def test_model_params_progress_callback_with_model(model_path):
    """Test that progress callback is actually called during model loading."""
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        return True  # continue loading

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    # Load model - this should trigger progress callbacks
    model = cy.LlamaModel(model_path, params)

    # Verify callback was called with progress values
    assert len(progress_values) > 0, "Progress callback should have been called"
    assert all(0.0 <= p <= 1.0 for p in progress_values), "Progress values should be between 0 and 1"
    # The final progress should be 1.0 (complete)
    assert progress_values[-1] == 1.0, "Final progress should be 1.0"


def test_model_params_progress_callback_abort(model_path):
    """Test that returning False from progress callback aborts loading."""
    progress_values = []

    def on_progress(progress: float) -> bool:
        progress_values.append(progress)
        # Abort after first callback
        return False

    params = cy.LlamaModelParams()
    params.progress_callback = on_progress

    # Loading should fail due to abort
    with pytest.raises(Exception):
        model = cy.LlamaModel(model_path, params)

    # Callback should have been called at least once
    assert len(progress_values) >= 1, "Progress callback should have been called before abort"


def test_default_context_params():
    params = cy.LlamaContextParams()
    assert params.n_ctx == 512
    assert params.n_batch == 2048
    assert params.n_ubatch == 512
    assert params.n_seq_max == 1
    assert params.n_threads == cy.GGML_DEFAULT_N_THREADS
    assert params.n_threads_batch == cy.GGML_DEFAULT_N_THREADS
    assert params.rope_scaling_type == -1  # LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
    assert params.pooling_type == -1  # LLAMA_POOLING_TYPE_UNSPECIFIED = -1
    assert params.attention_type == -1  # LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
    assert params.rope_freq_base == 0.0
    assert params.rope_freq_scale == 0.0
    assert params.yarn_ext_factor == -1.0
    assert params.yarn_attn_factor == approx(-1.0)
    assert params.yarn_beta_fast == -1.0
    assert params.yarn_beta_slow == -1.0
    assert params.yarn_orig_ctx == 0
    assert params.type_k == 1  # GGML_TYPE_F16 = 1
    assert params.type_v == 1  # GGML_TYPE_F16 = 1
    assert params.offload_kqv == True
    assert params.no_perf == True


def test_default_model_quantize_params():
    params = cy.LlamaModelQuantizeParams()
    assert params.nthread == 0
    assert params.ftype == 7  # LLAMA_FTYPE_MOSTLY_Q8_0 = 7
    assert params.output_tensor_type == 43  # GGML_TYPE_COUNT = 43
    assert params.token_embedding_type == 43  # GGML_TYPE_COUNT = 43
    assert params.allow_requantize == False
    assert params.quantize_output_tensor == True
    assert params.only_copy == False
    assert params.pure == False
    assert params.keep_split == False


def test_default_ggml_threadpool_params():
    params = cy.GgmlThreadPoolParams(n_threads=10)
    assert params.n_threads == 10
    assert params.prio == 0
    assert params.poll == 50
    assert params.strict_cpu == False
    assert params.paused == False
    assert params.cpumask == [False] * cy.GGML_MAX_N_THREADS
