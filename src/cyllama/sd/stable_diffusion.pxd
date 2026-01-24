# distutils: language = c++
# cython: language_level=3

"""
Cython declarations for stable-diffusion.cpp C API.
"""

from libc.stdint cimport int32_t, int64_t, uint8_t, uint32_t
from libc.stddef cimport size_t
from libcpp cimport bool as bint

cdef extern from "stable-diffusion.h":
    # =========================================================================
    # Enums
    # =========================================================================

    ctypedef enum rng_type_t:
        STD_DEFAULT_RNG
        CUDA_RNG
        CPU_RNG
        RNG_TYPE_COUNT

    ctypedef enum sample_method_t:
        EULER_SAMPLE_METHOD
        EULER_A_SAMPLE_METHOD
        HEUN_SAMPLE_METHOD
        DPM2_SAMPLE_METHOD
        DPMPP2S_A_SAMPLE_METHOD
        DPMPP2M_SAMPLE_METHOD
        DPMPP2Mv2_SAMPLE_METHOD
        IPNDM_SAMPLE_METHOD
        IPNDM_V_SAMPLE_METHOD
        LCM_SAMPLE_METHOD
        DDIM_TRAILING_SAMPLE_METHOD
        TCD_SAMPLE_METHOD
        SAMPLE_METHOD_COUNT

    ctypedef enum scheduler_t:
        DISCRETE_SCHEDULER
        KARRAS_SCHEDULER
        EXPONENTIAL_SCHEDULER
        AYS_SCHEDULER
        GITS_SCHEDULER
        SGM_UNIFORM_SCHEDULER
        SIMPLE_SCHEDULER
        SMOOTHSTEP_SCHEDULER
        LCM_SCHEDULER
        SCHEDULER_COUNT

    ctypedef enum prediction_t:
        EPS_PRED
        V_PRED
        EDM_V_PRED
        FLOW_PRED
        FLUX_FLOW_PRED
        FLUX2_FLOW_PRED
        PREDICTION_COUNT

    ctypedef enum sd_type_t:
        SD_TYPE_F32
        SD_TYPE_F16
        SD_TYPE_Q4_0
        SD_TYPE_Q4_1
        SD_TYPE_Q5_0
        SD_TYPE_Q5_1
        SD_TYPE_Q8_0
        SD_TYPE_Q8_1
        SD_TYPE_Q2_K
        SD_TYPE_Q3_K
        SD_TYPE_Q4_K
        SD_TYPE_Q5_K
        SD_TYPE_Q6_K
        SD_TYPE_Q8_K
        SD_TYPE_IQ2_XXS
        SD_TYPE_IQ2_XS
        SD_TYPE_IQ3_XXS
        SD_TYPE_IQ1_S
        SD_TYPE_IQ4_NL
        SD_TYPE_IQ3_S
        SD_TYPE_IQ2_S
        SD_TYPE_IQ4_XS
        SD_TYPE_I8
        SD_TYPE_I16
        SD_TYPE_I32
        SD_TYPE_I64
        SD_TYPE_F64
        SD_TYPE_IQ1_M
        SD_TYPE_BF16
        SD_TYPE_TQ1_0
        SD_TYPE_TQ2_0
        SD_TYPE_MXFP4
        SD_TYPE_COUNT

    ctypedef enum sd_log_level_t:
        SD_LOG_DEBUG
        SD_LOG_INFO
        SD_LOG_WARN
        SD_LOG_ERROR

    ctypedef enum preview_t:
        PREVIEW_NONE
        PREVIEW_PROJ
        PREVIEW_TAE
        PREVIEW_VAE
        PREVIEW_COUNT

    ctypedef enum lora_apply_mode_t:
        LORA_APPLY_AUTO
        LORA_APPLY_IMMEDIATELY
        LORA_APPLY_AT_RUNTIME
        LORA_APPLY_MODE_COUNT

    # =========================================================================
    # Structures
    # =========================================================================

    ctypedef struct sd_tiling_params_t:
        bint enabled
        int tile_size_x
        int tile_size_y
        float target_overlap
        float rel_size_x
        float rel_size_y

    ctypedef struct sd_embedding_t:
        const char* name
        const char* path

    ctypedef struct sd_lora_t:
        bint is_high_noise
        float multiplier
        const char* path

    ctypedef struct sd_ctx_params_t:
        const char* model_path
        const char* clip_l_path
        const char* clip_g_path
        const char* clip_vision_path
        const char* t5xxl_path
        const char* llm_path
        const char* llm_vision_path
        const char* diffusion_model_path
        const char* high_noise_diffusion_model_path
        const char* vae_path
        const char* taesd_path
        const char* control_net_path
        const sd_embedding_t* embeddings
        uint32_t embedding_count
        const char* photo_maker_path
        const char* tensor_type_rules
        bint vae_decode_only
        bint free_params_immediately
        int n_threads
        sd_type_t wtype
        rng_type_t rng_type
        rng_type_t sampler_rng_type
        prediction_t prediction
        lora_apply_mode_t lora_apply_mode
        bint offload_params_to_cpu
        bint keep_clip_on_cpu
        bint keep_control_net_on_cpu
        bint keep_vae_on_cpu
        bint diffusion_flash_attn
        bint tae_preview_only
        bint diffusion_conv_direct
        bint vae_conv_direct
        bint force_sdxl_vae_conv_scale
        bint chroma_use_dit_mask
        bint chroma_use_t5_mask
        int chroma_t5_mask_pad
        float flow_shift

    ctypedef struct sd_image_t:
        uint32_t width
        uint32_t height
        uint32_t channel
        uint8_t* data

    ctypedef struct sd_slg_params_t:
        int* layers
        size_t layer_count
        float layer_start
        float layer_end
        float scale

    ctypedef struct sd_guidance_params_t:
        float txt_cfg
        float img_cfg
        float distilled_guidance
        sd_slg_params_t slg

    ctypedef struct sd_sample_params_t:
        sd_guidance_params_t guidance
        scheduler_t scheduler
        sample_method_t sample_method
        int sample_steps
        float eta
        int shifted_timestep
        float* custom_sigmas
        int custom_sigmas_count

    ctypedef struct sd_pm_params_t:
        sd_image_t* id_images
        int id_images_count
        const char* id_embed_path
        float style_strength

    cdef enum sd_cache_mode_t:
        SD_CACHE_DISABLED = 0
        SD_CACHE_EASYCACHE
        SD_CACHE_UCACHE
        SD_CACHE_DBCACHE
        SD_CACHE_TAYLORSEER
        SD_CACHE_CACHE_DIT

    ctypedef struct sd_cache_params_t:
        sd_cache_mode_t mode
        float reuse_threshold
        float start_percent
        float end_percent
        float error_decay_rate
        bint use_relative_threshold
        bint reset_error_on_compute
        int Fn_compute_blocks
        int Bn_compute_blocks
        float residual_diff_threshold
        int max_warmup_steps
        int max_cached_steps
        int max_continuous_cached_steps
        int taylorseer_n_derivatives
        int taylorseer_skip_interval
        const char* scm_mask
        bint scm_policy_dynamic

    ctypedef struct sd_img_gen_params_t:
        const sd_lora_t* loras
        uint32_t lora_count
        const char* prompt
        const char* negative_prompt
        int clip_skip
        sd_image_t init_image
        sd_image_t* ref_images
        int ref_images_count
        bint auto_resize_ref_image
        bint increase_ref_index
        sd_image_t mask_image
        int width
        int height
        sd_sample_params_t sample_params
        float strength
        int64_t seed
        int batch_count
        sd_image_t control_image
        float control_strength
        sd_pm_params_t pm_params
        sd_tiling_params_t vae_tiling_params
        sd_cache_params_t cache

    ctypedef struct sd_vid_gen_params_t:
        const sd_lora_t* loras
        uint32_t lora_count
        const char* prompt
        const char* negative_prompt
        int clip_skip
        sd_image_t init_image
        sd_image_t end_image
        sd_image_t* control_frames
        int control_frames_size
        int width
        int height
        sd_sample_params_t sample_params
        sd_sample_params_t high_noise_sample_params
        float moe_boundary
        float strength
        int64_t seed
        int video_frames
        float vace_strength
        sd_tiling_params_t vae_tiling_params
        sd_cache_params_t cache

    # Opaque context types
    ctypedef struct sd_ctx_t:
        pass

    ctypedef struct upscaler_ctx_t:
        pass

    # =========================================================================
    # Callback types
    # =========================================================================

    ctypedef void (*sd_log_cb_t)(sd_log_level_t level, const char* text, void* data)
    ctypedef void (*sd_progress_cb_t)(int step, int steps, float time, void* data)
    ctypedef void (*sd_preview_cb_t)(int step, int frame_count, sd_image_t* frames, bint is_noisy, void* data)

    # =========================================================================
    # Functions - Callbacks
    # =========================================================================

    void sd_set_log_callback(sd_log_cb_t sd_log_cb, void* data)
    void sd_set_progress_callback(sd_progress_cb_t cb, void* data)
    void sd_set_preview_callback(sd_preview_cb_t cb, preview_t mode, int interval, bint denoised, bint noisy, void* data)

    # =========================================================================
    # Functions - System info
    # =========================================================================

    int32_t sd_get_num_physical_cores()
    const char* sd_get_system_info()

    # =========================================================================
    # Functions - Type/enum name conversions
    # =========================================================================

    const char* sd_type_name(sd_type_t type)
    sd_type_t str_to_sd_type(const char* str)
    const char* sd_rng_type_name(rng_type_t rng_type)
    rng_type_t str_to_rng_type(const char* str)
    const char* sd_sample_method_name(sample_method_t sample_method)
    sample_method_t str_to_sample_method(const char* str)
    const char* sd_scheduler_name(scheduler_t scheduler)
    scheduler_t str_to_scheduler(const char* str)
    const char* sd_prediction_name(prediction_t prediction)
    prediction_t str_to_prediction(const char* str)
    const char* sd_preview_name(preview_t preview)
    preview_t str_to_preview(const char* str)
    const char* sd_lora_apply_mode_name(lora_apply_mode_t mode)
    lora_apply_mode_t str_to_lora_apply_mode(const char* str)

    # =========================================================================
    # Functions - Parameter initialization
    # =========================================================================

    void sd_cache_params_init(sd_cache_params_t* cache_params)
    void sd_ctx_params_init(sd_ctx_params_t* sd_ctx_params)
    char* sd_ctx_params_to_str(const sd_ctx_params_t* sd_ctx_params)
    void sd_sample_params_init(sd_sample_params_t* sample_params)
    char* sd_sample_params_to_str(const sd_sample_params_t* sample_params)
    void sd_img_gen_params_init(sd_img_gen_params_t* sd_img_gen_params)
    char* sd_img_gen_params_to_str(const sd_img_gen_params_t* sd_img_gen_params)
    void sd_vid_gen_params_init(sd_vid_gen_params_t* sd_vid_gen_params)

    # =========================================================================
    # Functions - Context management
    # =========================================================================

    sd_ctx_t* new_sd_ctx(const sd_ctx_params_t* sd_ctx_params)
    void free_sd_ctx(sd_ctx_t* sd_ctx)

    sample_method_t sd_get_default_sample_method(const sd_ctx_t* sd_ctx)
    scheduler_t sd_get_default_scheduler(const sd_ctx_t* sd_ctx, sample_method_t sample_method)

    # =========================================================================
    # Functions - Image generation
    # =========================================================================

    sd_image_t* generate_image(sd_ctx_t* sd_ctx, const sd_img_gen_params_t* sd_img_gen_params)

    # =========================================================================
    # Functions - Video generation
    # =========================================================================

    sd_image_t* generate_video(sd_ctx_t* sd_ctx, const sd_vid_gen_params_t* sd_vid_gen_params, int* num_frames_out)

    # =========================================================================
    # Functions - Upscaling
    # =========================================================================

    upscaler_ctx_t* new_upscaler_ctx(const char* esrgan_path,
                                      bint offload_params_to_cpu,
                                      bint direct,
                                      int n_threads,
                                      int tile_size)
    void free_upscaler_ctx(upscaler_ctx_t* upscaler_ctx)
    sd_image_t upscale(upscaler_ctx_t* upscaler_ctx, sd_image_t input_image, uint32_t upscale_factor)
    int get_upscale_factor(upscaler_ctx_t* upscaler_ctx)

    # =========================================================================
    # Functions - Model conversion
    # =========================================================================

    bint convert(const char* input_path,
                 const char* vae_path,
                 const char* output_path,
                 sd_type_t output_type,
                 const char* tensor_type_rules,
                 bint convert_name)

    # =========================================================================
    # Functions - Preprocessing
    # =========================================================================

    bint preprocess_canny(sd_image_t image,
                          float high_threshold,
                          float low_threshold,
                          float weak,
                          float strong,
                          bint inverse)

    # =========================================================================
    # Functions - Version info
    # =========================================================================

    const char* sd_commit()
    const char* sd_version()
