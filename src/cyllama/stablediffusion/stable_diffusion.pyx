# distutils: language = c++
# cython: language_level=3

"""
Cython wrapper for stable-diffusion.cpp.

Provides Python bindings for image generation using Stable Diffusion models.
"""

import os
from typing import Optional, List, Callable, Union
from enum import IntEnum

cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.stdint cimport uint8_t, uint32_t, int64_t

from .stable_diffusion cimport *

# Try to import numpy - optional but recommended
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Try to import PIL - optional
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Python Enums (mirror C enums)
# =============================================================================

class RngType(IntEnum):
    """Random number generator types."""
    STD_DEFAULT = STD_DEFAULT_RNG
    CUDA = CUDA_RNG
    CPU = CPU_RNG


class SampleMethod(IntEnum):
    """Sampling methods for diffusion."""
    EULER = EULER_SAMPLE_METHOD
    EULER_A = EULER_A_SAMPLE_METHOD
    HEUN = HEUN_SAMPLE_METHOD
    DPM2 = DPM2_SAMPLE_METHOD
    DPMPP2S_A = DPMPP2S_A_SAMPLE_METHOD
    DPMPP2M = DPMPP2M_SAMPLE_METHOD
    DPMPP2Mv2 = DPMPP2Mv2_SAMPLE_METHOD
    IPNDM = IPNDM_SAMPLE_METHOD
    IPNDM_V = IPNDM_V_SAMPLE_METHOD
    LCM = LCM_SAMPLE_METHOD
    DDIM_TRAILING = DDIM_TRAILING_SAMPLE_METHOD
    TCD = TCD_SAMPLE_METHOD


class Scheduler(IntEnum):
    """Noise schedulers."""
    DISCRETE = DISCRETE_SCHEDULER
    KARRAS = KARRAS_SCHEDULER
    EXPONENTIAL = EXPONENTIAL_SCHEDULER
    AYS = AYS_SCHEDULER
    GITS = GITS_SCHEDULER
    SGM_UNIFORM = SGM_UNIFORM_SCHEDULER
    SIMPLE = SIMPLE_SCHEDULER
    SMOOTHSTEP = SMOOTHSTEP_SCHEDULER
    LCM = LCM_SCHEDULER


class Prediction(IntEnum):
    """Prediction types."""
    DEFAULT = DEFAULT_PRED
    EPS = EPS_PRED
    V = V_PRED
    EDM_V = EDM_V_PRED
    SD3_FLOW = SD3_FLOW_PRED
    FLUX_FLOW = FLUX_FLOW_PRED


class SDType(IntEnum):
    """Data types for model weights."""
    F32 = SD_TYPE_F32
    F16 = SD_TYPE_F16
    Q4_0 = SD_TYPE_Q4_0
    Q4_1 = SD_TYPE_Q4_1
    Q5_0 = SD_TYPE_Q5_0
    Q5_1 = SD_TYPE_Q5_1
    Q8_0 = SD_TYPE_Q8_0
    Q8_1 = SD_TYPE_Q8_1
    Q2_K = SD_TYPE_Q2_K
    Q3_K = SD_TYPE_Q3_K
    Q4_K = SD_TYPE_Q4_K
    Q5_K = SD_TYPE_Q5_K
    Q6_K = SD_TYPE_Q6_K
    Q8_K = SD_TYPE_Q8_K
    BF16 = SD_TYPE_BF16


class LogLevel(IntEnum):
    """Log levels."""
    DEBUG = SD_LOG_DEBUG
    INFO = SD_LOG_INFO
    WARN = SD_LOG_WARN
    ERROR = SD_LOG_ERROR


class PreviewMode(IntEnum):
    """Preview modes during generation."""
    NONE = PREVIEW_NONE
    PROJ = PREVIEW_PROJ
    TAE = PREVIEW_TAE
    VAE = PREVIEW_VAE


class LoraApplyMode(IntEnum):
    """LoRA application modes."""
    AUTO = LORA_APPLY_AUTO
    IMMEDIATELY = LORA_APPLY_IMMEDIATELY
    AT_RUNTIME = LORA_APPLY_AT_RUNTIME


# =============================================================================
# Utility functions
# =============================================================================

def get_num_cores() -> int:
    """Get the number of physical CPU cores."""
    return get_num_physical_cores()


def get_system_info() -> str:
    """Get system information string."""
    cdef const char* info = sd_get_system_info()
    return info.decode('utf-8') if info else ""


def type_name(sd_type: SDType) -> str:
    """Get the name of an SD type."""
    cdef const char* name = sd_type_name(<sd_type_t>sd_type)
    return name.decode('utf-8') if name else ""


def sample_method_name(method: SampleMethod) -> str:
    """Get the name of a sample method."""
    cdef const char* name = sd_sample_method_name(<sample_method_t>method)
    return name.decode('utf-8') if name else ""


def scheduler_name(sched: Scheduler) -> str:
    """Get the name of a scheduler."""
    cdef const char* name = sd_scheduler_name(<scheduler_t>sched)
    return name.decode('utf-8') if name else ""


# =============================================================================
# Callback management
# =============================================================================

# Global callback storage (prevent garbage collection)
cdef object _log_callback = None
cdef object _progress_callback = None


cdef void _c_log_callback(sd_log_level_t level, const char* text, void* data) noexcept with gil:
    """C callback that forwards to Python log callback."""
    global _log_callback
    if _log_callback is not None:
        try:
            py_text = text.decode('utf-8') if text else ""
            _log_callback(LogLevel(level), py_text)
        except Exception:
            pass  # Ignore exceptions in callbacks


cdef void _c_progress_callback(int step, int steps, float time, void* data) noexcept with gil:
    """C callback that forwards to Python progress callback."""
    global _progress_callback
    if _progress_callback is not None:
        try:
            _progress_callback(step, steps, time)
        except Exception:
            pass  # Ignore exceptions in callbacks


def set_log_callback(callback: Optional[Callable[[LogLevel, str], None]]):
    """
    Set the logging callback function.

    Args:
        callback: Function that receives (level, message) or None to disable
    """
    global _log_callback
    _log_callback = callback
    if callback is not None:
        sd_set_log_callback(_c_log_callback, NULL)
    else:
        sd_set_log_callback(NULL, NULL)


def set_progress_callback(callback: Optional[Callable[[int, int, float], None]]):
    """
    Set the progress callback function.

    Args:
        callback: Function that receives (step, total_steps, time) or None to disable
    """
    global _progress_callback
    _progress_callback = callback
    if callback is not None:
        sd_set_progress_callback(_c_progress_callback, NULL)
    else:
        sd_set_progress_callback(NULL, NULL)


# =============================================================================
# SDImage class
# =============================================================================

cdef class SDImage:
    """
    Wrapper for sd_image_t representing an image.

    Provides conversion to/from numpy arrays and PIL Images.
    """
    cdef sd_image_t _image
    cdef bint _owns_data

    def __cinit__(self):
        memset(&self._image, 0, sizeof(sd_image_t))
        self._owns_data = False

    def __dealloc__(self):
        if self._owns_data and self._image.data != NULL:
            free(self._image.data)
            self._image.data = NULL

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._image.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._image.height

    @property
    def channels(self) -> int:
        """Number of color channels (typically 3 for RGB)."""
        return self._image.channel

    @property
    def shape(self) -> tuple:
        """Image shape as (height, width, channels)."""
        return (self._image.height, self._image.width, self._image.channel)

    @property
    def size(self) -> int:
        """Total size in bytes."""
        return self._image.width * self._image.height * self._image.channel

    @property
    def is_valid(self) -> bool:
        """Check if image has valid data."""
        return self._image.data != NULL and self._image.width > 0 and self._image.height > 0

    def to_numpy(self):
        """
        Convert to numpy array.

        Returns:
            numpy.ndarray: Image data as (H, W, C) uint8 array

        Raises:
            ImportError: If numpy is not installed
            ValueError: If image has no data
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for to_numpy()")
        if not self.is_valid:
            raise ValueError("Image has no valid data")

        cdef size_t size = self._image.width * self._image.height * self._image.channel
        arr = np.empty((self._image.height, self._image.width, self._image.channel), dtype=np.uint8)
        cdef uint8_t[:, :, :] arr_view = arr
        memcpy(&arr_view[0, 0, 0], self._image.data, size)
        return arr

    def to_pil(self):
        """
        Convert to PIL Image.

        Returns:
            PIL.Image: Image object

        Raises:
            ImportError: If PIL/Pillow is not installed
            ValueError: If image has no data
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for to_pil()")
        arr = self.to_numpy()
        if self._image.channel == 3:
            return PILImage.fromarray(arr, mode='RGB')
        elif self._image.channel == 4:
            return PILImage.fromarray(arr, mode='RGBA')
        elif self._image.channel == 1:
            return PILImage.fromarray(arr[:, :, 0], mode='L')
        else:
            return PILImage.fromarray(arr)

    def save(self, path: str):
        """
        Save image to file using PIL.

        Args:
            path: Output file path (format determined by extension)
        """
        img = self.to_pil()
        img.save(path)

    @staticmethod
    def from_numpy(arr) -> "SDImage":
        """
        Create SDImage from numpy array.

        Args:
            arr: numpy array of shape (H, W, C) or (H, W) with dtype uint8

        Returns:
            SDImage: New image object
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for from_numpy()")

        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]

        cdef SDImage img = SDImage()
        img._image.height = arr.shape[0]
        img._image.width = arr.shape[1]
        img._image.channel = arr.shape[2]

        cdef size_t size = arr.shape[0] * arr.shape[1] * arr.shape[2]
        img._image.data = <uint8_t*>malloc(size)
        if img._image.data == NULL:
            raise MemoryError("Failed to allocate image data")

        cdef uint8_t[:, :, :] arr_view = arr
        memcpy(img._image.data, &arr_view[0, 0, 0], size)
        img._owns_data = True
        return img

    @staticmethod
    def from_pil(pil_image) -> "SDImage":
        """
        Create SDImage from PIL Image.

        Args:
            pil_image: PIL Image object

        Returns:
            SDImage: New image object
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for from_pil()")
        if not HAS_NUMPY:
            raise ImportError("numpy is required for from_pil()")

        # Convert to RGB if necessary
        if pil_image.mode not in ('RGB', 'RGBA', 'L'):
            pil_image = pil_image.convert('RGB')

        arr = np.array(pil_image)
        return SDImage.from_numpy(arr)

    @staticmethod
    def load(path: str) -> "SDImage":
        """
        Load image from file using PIL.

        Args:
            path: Input file path

        Returns:
            SDImage: Loaded image
        """
        if not HAS_PIL:
            raise ImportError("PIL/Pillow is required for load()")
        pil_image = PILImage.open(path)
        return SDImage.from_pil(pil_image)

    @staticmethod
    cdef SDImage _from_c_image(sd_image_t c_image, bint owns_data=True):
        """Create SDImage from C struct (internal use)."""
        cdef SDImage img = SDImage()
        img._image = c_image
        img._owns_data = owns_data
        return img


# =============================================================================
# SDContextParams class
# =============================================================================

cdef class SDContextParams:
    """
    Parameters for creating an SDContext.

    Wraps sd_ctx_params_t with Python-friendly property access.
    """
    cdef sd_ctx_params_t _params

    # Store bytes objects to keep strings alive
    cdef bytes _model_path_bytes
    cdef bytes _clip_l_path_bytes
    cdef bytes _clip_g_path_bytes
    cdef bytes _clip_vision_path_bytes
    cdef bytes _t5xxl_path_bytes
    cdef bytes _qwen2vl_path_bytes
    cdef bytes _qwen2vl_vision_path_bytes
    cdef bytes _diffusion_model_path_bytes
    cdef bytes _high_noise_diffusion_model_path_bytes
    cdef bytes _vae_path_bytes
    cdef bytes _taesd_path_bytes
    cdef bytes _control_net_path_bytes
    cdef bytes _lora_model_dir_bytes
    cdef bytes _embedding_dir_bytes
    cdef bytes _photo_maker_path_bytes
    cdef bytes _tensor_type_rules_bytes

    def __cinit__(self):
        sd_ctx_params_init(&self._params)

    def __init__(self,
                 model_path: Optional[str] = None,
                 vae_path: Optional[str] = None,
                 clip_l_path: Optional[str] = None,
                 clip_g_path: Optional[str] = None,
                 t5xxl_path: Optional[str] = None,
                 diffusion_model_path: Optional[str] = None,
                 lora_model_dir: Optional[str] = None,
                 embedding_dir: Optional[str] = None,
                 n_threads: int = -1,
                 wtype: SDType = SDType.F16,
                 vae_decode_only: bool = True):
        """
        Initialize context parameters.

        Args:
            model_path: Path to main model file (.safetensors, .ckpt, .gguf)
            vae_path: Path to VAE model file (optional)
            clip_l_path: Path to CLIP-L model (for SDXL/SD3)
            clip_g_path: Path to CLIP-G model (for SDXL/SD3)
            t5xxl_path: Path to T5-XXL model (for SD3/FLUX)
            diffusion_model_path: Path to diffusion model (for split models)
            lora_model_dir: Directory containing LoRA files
            embedding_dir: Directory containing embedding files
            n_threads: Number of threads (-1 for auto)
            wtype: Weight type for computation
            vae_decode_only: Only decode VAE (faster if not doing img2img)
        """
        if model_path:
            self.model_path = model_path
        if vae_path:
            self.vae_path = vae_path
        if clip_l_path:
            self.clip_l_path = clip_l_path
        if clip_g_path:
            self.clip_g_path = clip_g_path
        if t5xxl_path:
            self.t5xxl_path = t5xxl_path
        if diffusion_model_path:
            self.diffusion_model_path = diffusion_model_path
        if lora_model_dir:
            self.lora_model_dir = lora_model_dir
        if embedding_dir:
            self.embedding_dir = embedding_dir
        if n_threads > 0:
            self.n_threads = n_threads
        self.wtype = wtype
        self.vae_decode_only = vae_decode_only

    # --- Model paths ---

    @property
    def model_path(self) -> Optional[str]:
        """Path to main model file."""
        if self._params.model_path:
            return self._params.model_path.decode('utf-8')
        return None

    @model_path.setter
    def model_path(self, value: Optional[str]):
        if value:
            self._model_path_bytes = value.encode('utf-8')
            self._params.model_path = self._model_path_bytes
        else:
            self._params.model_path = NULL

    @property
    def vae_path(self) -> Optional[str]:
        """Path to VAE model file."""
        if self._params.vae_path:
            return self._params.vae_path.decode('utf-8')
        return None

    @vae_path.setter
    def vae_path(self, value: Optional[str]):
        if value:
            self._vae_path_bytes = value.encode('utf-8')
            self._params.vae_path = self._vae_path_bytes
        else:
            self._params.vae_path = NULL

    @property
    def clip_l_path(self) -> Optional[str]:
        """Path to CLIP-L model."""
        if self._params.clip_l_path:
            return self._params.clip_l_path.decode('utf-8')
        return None

    @clip_l_path.setter
    def clip_l_path(self, value: Optional[str]):
        if value:
            self._clip_l_path_bytes = value.encode('utf-8')
            self._params.clip_l_path = self._clip_l_path_bytes
        else:
            self._params.clip_l_path = NULL

    @property
    def clip_g_path(self) -> Optional[str]:
        """Path to CLIP-G model."""
        if self._params.clip_g_path:
            return self._params.clip_g_path.decode('utf-8')
        return None

    @clip_g_path.setter
    def clip_g_path(self, value: Optional[str]):
        if value:
            self._clip_g_path_bytes = value.encode('utf-8')
            self._params.clip_g_path = self._clip_g_path_bytes
        else:
            self._params.clip_g_path = NULL

    @property
    def t5xxl_path(self) -> Optional[str]:
        """Path to T5-XXL model."""
        if self._params.t5xxl_path:
            return self._params.t5xxl_path.decode('utf-8')
        return None

    @t5xxl_path.setter
    def t5xxl_path(self, value: Optional[str]):
        if value:
            self._t5xxl_path_bytes = value.encode('utf-8')
            self._params.t5xxl_path = self._t5xxl_path_bytes
        else:
            self._params.t5xxl_path = NULL

    @property
    def diffusion_model_path(self) -> Optional[str]:
        """Path to diffusion model."""
        if self._params.diffusion_model_path:
            return self._params.diffusion_model_path.decode('utf-8')
        return None

    @diffusion_model_path.setter
    def diffusion_model_path(self, value: Optional[str]):
        if value:
            self._diffusion_model_path_bytes = value.encode('utf-8')
            self._params.diffusion_model_path = self._diffusion_model_path_bytes
        else:
            self._params.diffusion_model_path = NULL

    @property
    def lora_model_dir(self) -> Optional[str]:
        """Directory containing LoRA files."""
        if self._params.lora_model_dir:
            return self._params.lora_model_dir.decode('utf-8')
        return None

    @lora_model_dir.setter
    def lora_model_dir(self, value: Optional[str]):
        if value:
            self._lora_model_dir_bytes = value.encode('utf-8')
            self._params.lora_model_dir = self._lora_model_dir_bytes
        else:
            self._params.lora_model_dir = NULL

    @property
    def embedding_dir(self) -> Optional[str]:
        """Directory containing embedding files."""
        if self._params.embedding_dir:
            return self._params.embedding_dir.decode('utf-8')
        return None

    @embedding_dir.setter
    def embedding_dir(self, value: Optional[str]):
        if value:
            self._embedding_dir_bytes = value.encode('utf-8')
            self._params.embedding_dir = self._embedding_dir_bytes
        else:
            self._params.embedding_dir = NULL

    # --- Numeric/enum parameters ---

    @property
    def n_threads(self) -> int:
        """Number of threads."""
        return self._params.n_threads

    @n_threads.setter
    def n_threads(self, value: int):
        self._params.n_threads = value

    @property
    def wtype(self) -> SDType:
        """Weight type."""
        return SDType(self._params.wtype)

    @wtype.setter
    def wtype(self, value: SDType):
        self._params.wtype = <sd_type_t>value

    @property
    def rng_type(self) -> RngType:
        """RNG type."""
        return RngType(self._params.rng_type)

    @rng_type.setter
    def rng_type(self, value: RngType):
        self._params.rng_type = <rng_type_t>value

    @property
    def vae_decode_only(self) -> bool:
        """Only decode VAE (faster if not doing img2img)."""
        return self._params.vae_decode_only

    @vae_decode_only.setter
    def vae_decode_only(self, value: bool):
        self._params.vae_decode_only = value

    @property
    def diffusion_flash_attn(self) -> bool:
        """Use flash attention in diffusion model."""
        return self._params.diffusion_flash_attn

    @diffusion_flash_attn.setter
    def diffusion_flash_attn(self, value: bool):
        self._params.diffusion_flash_attn = value

    @property
    def offload_params_to_cpu(self) -> bool:
        """Offload parameters to CPU."""
        return self._params.offload_params_to_cpu

    @offload_params_to_cpu.setter
    def offload_params_to_cpu(self, value: bool):
        self._params.offload_params_to_cpu = value

    def __str__(self) -> str:
        """Get string representation of parameters."""
        cdef char* s = sd_ctx_params_to_str(&self._params)
        if s:
            result = s.decode('utf-8')
            free(s)
            return result
        return "SDContextParams()"


# =============================================================================
# SDSampleParams class
# =============================================================================

cdef class SDSampleParams:
    """
    Sampling parameters for image generation.
    """
    cdef sd_sample_params_t _params

    def __cinit__(self):
        sd_sample_params_init(&self._params)

    def __init__(self,
                 sample_method: SampleMethod = SampleMethod.EULER_A,
                 scheduler: Scheduler = Scheduler.DISCRETE,
                 sample_steps: int = 20,
                 cfg_scale: float = 7.0,
                 eta: float = 0.0):
        """
        Initialize sampling parameters.

        Args:
            sample_method: Sampling method to use
            scheduler: Noise scheduler
            sample_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            eta: Eta parameter for some samplers
        """
        self.sample_method = sample_method
        self.scheduler = scheduler
        self.sample_steps = sample_steps
        self.cfg_scale = cfg_scale
        self.eta = eta

    @property
    def sample_method(self) -> SampleMethod:
        """Sampling method."""
        return SampleMethod(self._params.sample_method)

    @sample_method.setter
    def sample_method(self, value: SampleMethod):
        self._params.sample_method = <sample_method_t>value

    @property
    def scheduler(self) -> Scheduler:
        """Noise scheduler."""
        return Scheduler(self._params.scheduler)

    @scheduler.setter
    def scheduler(self, value: Scheduler):
        self._params.scheduler = <scheduler_t>value

    @property
    def sample_steps(self) -> int:
        """Number of sampling steps."""
        return self._params.sample_steps

    @sample_steps.setter
    def sample_steps(self, value: int):
        self._params.sample_steps = value

    @property
    def cfg_scale(self) -> float:
        """Classifier-free guidance scale."""
        return self._params.guidance.txt_cfg

    @cfg_scale.setter
    def cfg_scale(self, value: float):
        self._params.guidance.txt_cfg = value

    @property
    def eta(self) -> float:
        """Eta parameter."""
        return self._params.eta

    @eta.setter
    def eta(self, value: float):
        self._params.eta = value

    def __str__(self) -> str:
        """Get string representation."""
        cdef char* s = sd_sample_params_to_str(&self._params)
        if s:
            result = s.decode('utf-8')
            free(s)
            return result
        return "SDSampleParams()"


# =============================================================================
# SDImageGenParams class
# =============================================================================

cdef class SDImageGenParams:
    """
    Parameters for image generation.
    """
    cdef sd_img_gen_params_t _params
    cdef bytes _prompt_bytes
    cdef bytes _negative_prompt_bytes
    cdef SDSampleParams _sample_params
    cdef SDImage _init_image
    cdef SDImage _mask_image
    cdef SDImage _control_image

    def __cinit__(self):
        sd_img_gen_params_init(&self._params)
        self._sample_params = SDSampleParams()

    def __init__(self,
                 prompt: str = "",
                 negative_prompt: str = "",
                 width: int = 512,
                 height: int = 512,
                 seed: int = -1,
                 batch_count: int = 1,
                 sample_steps: int = 20,
                 cfg_scale: float = 7.0,
                 sample_method: SampleMethod = SampleMethod.EULER_A,
                 scheduler: Scheduler = Scheduler.DISCRETE,
                 strength: float = 0.75,
                 clip_skip: int = -1):
        """
        Initialize image generation parameters.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            width: Output image width
            height: Output image height
            seed: Random seed (-1 for random)
            batch_count: Number of images to generate
            sample_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            sample_method: Sampling method
            scheduler: Noise scheduler
            strength: Strength for img2img (0.0-1.0)
            clip_skip: Number of CLIP layers to skip
        """
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.width = width
        self.height = height
        self.seed = seed
        self.batch_count = batch_count
        self.strength = strength
        self.clip_skip = clip_skip

        # Set sample params
        self._sample_params.sample_steps = sample_steps
        self._sample_params.cfg_scale = cfg_scale
        self._sample_params.sample_method = sample_method
        self._sample_params.scheduler = scheduler
        self._params.sample_params = self._sample_params._params

    @property
    def prompt(self) -> str:
        """Text prompt."""
        if self._params.prompt:
            return self._params.prompt.decode('utf-8')
        return ""

    @prompt.setter
    def prompt(self, value: str):
        self._prompt_bytes = value.encode('utf-8')
        self._params.prompt = self._prompt_bytes

    @property
    def negative_prompt(self) -> str:
        """Negative prompt."""
        if self._params.negative_prompt:
            return self._params.negative_prompt.decode('utf-8')
        return ""

    @negative_prompt.setter
    def negative_prompt(self, value: str):
        self._negative_prompt_bytes = value.encode('utf-8')
        self._params.negative_prompt = self._negative_prompt_bytes

    @property
    def width(self) -> int:
        """Output width."""
        return self._params.width

    @width.setter
    def width(self, value: int):
        self._params.width = value

    @property
    def height(self) -> int:
        """Output height."""
        return self._params.height

    @height.setter
    def height(self, value: int):
        self._params.height = value

    @property
    def seed(self) -> int:
        """Random seed."""
        return self._params.seed

    @seed.setter
    def seed(self, value: int):
        self._params.seed = value

    @property
    def batch_count(self) -> int:
        """Number of images to generate."""
        return self._params.batch_count

    @batch_count.setter
    def batch_count(self, value: int):
        self._params.batch_count = value

    @property
    def strength(self) -> float:
        """Img2img strength."""
        return self._params.strength

    @strength.setter
    def strength(self, value: float):
        self._params.strength = value

    @property
    def clip_skip(self) -> int:
        """CLIP skip layers."""
        return self._params.clip_skip

    @clip_skip.setter
    def clip_skip(self, value: int):
        self._params.clip_skip = value

    @property
    def sample_params(self) -> SDSampleParams:
        """Sampling parameters."""
        return self._sample_params

    @sample_params.setter
    def sample_params(self, value: SDSampleParams):
        self._sample_params = value
        self._params.sample_params = value._params

    def set_init_image(self, image: SDImage):
        """Set the initial image for img2img."""
        self._init_image = image
        self._params.init_image = image._image

    def set_mask_image(self, image: SDImage):
        """Set the mask image for inpainting."""
        self._mask_image = image
        self._params.mask_image = image._image

    def set_control_image(self, image: SDImage, strength: float = 1.0):
        """Set the control image for ControlNet."""
        self._control_image = image
        self._params.control_image = image._image
        self._params.control_strength = strength

    def __str__(self) -> str:
        """Get string representation."""
        cdef char* s = sd_img_gen_params_to_str(&self._params)
        if s:
            result = s.decode('utf-8')
            free(s)
            return result
        return "SDImageGenParams()"


# =============================================================================
# SDContext class
# =============================================================================

cdef class SDContext:
    """
    Main Stable Diffusion context for image generation.

    Example:
        params = SDContextParams(model_path="sd-v1-5.safetensors")
        ctx = SDContext(params)
        images = ctx.generate("a photo of a cat")
    """
    cdef sd_ctx_t* _ctx
    cdef SDContextParams _params

    def __cinit__(self):
        self._ctx = NULL

    def __dealloc__(self):
        if self._ctx != NULL:
            free_sd_ctx(self._ctx)
            self._ctx = NULL

    def __init__(self, params: SDContextParams):
        """
        Create a new Stable Diffusion context.

        Args:
            params: Context parameters including model paths

        Raises:
            RuntimeError: If context creation fails
        """
        self._params = params
        self._ctx = new_sd_ctx(&params._params)
        if self._ctx == NULL:
            raise RuntimeError("Failed to create Stable Diffusion context. Check model path and parameters.")

    @property
    def is_valid(self) -> bool:
        """Check if context is valid."""
        return self._ctx != NULL

    def get_default_sample_method(self) -> SampleMethod:
        """Get the default sampling method for the loaded model."""
        if self._ctx == NULL:
            raise RuntimeError("Context not initialized")
        return SampleMethod(sd_get_default_sample_method(self._ctx))

    def get_default_scheduler(self) -> Scheduler:
        """Get the default scheduler for the loaded model."""
        if self._ctx == NULL:
            raise RuntimeError("Context not initialized")
        return Scheduler(sd_get_default_scheduler(self._ctx))

    def generate(self,
                 prompt: str,
                 negative_prompt: str = "",
                 width: int = 512,
                 height: int = 512,
                 seed: int = -1,
                 batch_count: int = 1,
                 sample_steps: int = 20,
                 cfg_scale: float = 7.0,
                 sample_method: Optional[SampleMethod] = None,
                 scheduler: Optional[Scheduler] = None,
                 init_image: Optional[SDImage] = None,
                 strength: float = 0.75,
                 clip_skip: int = -1) -> List[SDImage]:
        """
        Generate images from a text prompt.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            width: Output image width
            height: Output image height
            seed: Random seed (-1 for random)
            batch_count: Number of images to generate
            sample_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            sample_method: Sampling method (None for model default)
            scheduler: Noise scheduler (None for model default)
            init_image: Initial image for img2img (None for txt2img)
            strength: Img2img strength (0.0-1.0)
            clip_skip: Number of CLIP layers to skip

        Returns:
            List of generated SDImage objects

        Raises:
            RuntimeError: If generation fails
        """
        if self._ctx == NULL:
            raise RuntimeError("Context not initialized")

        # Use model defaults if not specified
        if sample_method is None:
            sample_method = self.get_default_sample_method()
        if scheduler is None:
            scheduler = self.get_default_scheduler()

        # Create generation parameters
        cdef SDImageGenParams gen_params = SDImageGenParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            batch_count=batch_count,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            strength=strength,
            clip_skip=clip_skip
        )

        # Set init image for img2img
        if init_image is not None:
            gen_params.set_init_image(init_image)

        return self.generate_with_params(gen_params)

    def generate_with_params(self, params: SDImageGenParams) -> List[SDImage]:
        """
        Generate images with full parameter control.

        Args:
            params: Image generation parameters

        Returns:
            List of generated SDImage objects

        Raises:
            RuntimeError: If generation fails
        """
        if self._ctx == NULL:
            raise RuntimeError("Context not initialized")

        # Ensure sample params are synced
        params._params.sample_params = params._sample_params._params

        cdef sd_image_t* result = generate_image(self._ctx, &params._params)
        if result == NULL:
            raise RuntimeError("Image generation failed")

        # Convert results to Python list
        images = []
        cdef int i
        for i in range(params.batch_count):
            img = SDImage._from_c_image(result[i], owns_data=True)
            images.append(img)

        # Free the array (but not individual image data, now owned by SDImage objects)
        free(result)

        return images

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        if self._ctx != NULL:
            free_sd_ctx(self._ctx)
            self._ctx = NULL


# =============================================================================
# Convenience functions
# =============================================================================

def text_to_image(
    model_path: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 512,
    height: int = 512,
    seed: int = -1,
    batch_count: int = 1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    sample_method: SampleMethod = SampleMethod.EULER_A,
    scheduler: Scheduler = Scheduler.DISCRETE,
    n_threads: int = -1,
    vae_path: Optional[str] = None,
    clip_l_path: Optional[str] = None,
    clip_g_path: Optional[str] = None,
    t5xxl_path: Optional[str] = None,
    lora_model_dir: Optional[str] = None,
    clip_skip: int = -1
) -> List[SDImage]:
    """
    Generate images from text prompt (convenience function).

    This creates a context, generates images, and cleans up automatically.
    For generating multiple images with the same model, use SDContext directly.

    Args:
        model_path: Path to model file
        prompt: Text prompt
        negative_prompt: Negative prompt
        width: Output width
        height: Output height
        seed: Random seed (-1 for random)
        batch_count: Number of images
        sample_steps: Diffusion steps
        cfg_scale: CFG scale
        sample_method: Sampling method
        scheduler: Noise scheduler
        n_threads: Number of threads
        vae_path: Path to VAE (optional)
        clip_l_path: Path to CLIP-L (for SDXL/SD3)
        clip_g_path: Path to CLIP-G (for SDXL/SD3)
        t5xxl_path: Path to T5-XXL (for SD3/FLUX)
        lora_model_dir: LoRA directory
        clip_skip: CLIP skip layers

    Returns:
        List of generated SDImage objects
    """
    params = SDContextParams(
        model_path=model_path,
        vae_path=vae_path,
        clip_l_path=clip_l_path,
        clip_g_path=clip_g_path,
        t5xxl_path=t5xxl_path,
        lora_model_dir=lora_model_dir,
        n_threads=n_threads
    )

    with SDContext(params) as ctx:
        return ctx.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            batch_count=batch_count,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            clip_skip=clip_skip
        )


def image_to_image(
    model_path: str,
    init_image: Union[SDImage, str],
    prompt: str,
    negative_prompt: str = "",
    strength: float = 0.75,
    seed: int = -1,
    sample_steps: int = 20,
    cfg_scale: float = 7.0,
    sample_method: SampleMethod = SampleMethod.EULER_A,
    scheduler: Scheduler = Scheduler.DISCRETE,
    n_threads: int = -1,
    vae_path: Optional[str] = None,
    clip_skip: int = -1
) -> List[SDImage]:
    """
    Generate images from an initial image (img2img).

    Args:
        model_path: Path to model file
        init_image: Initial image (SDImage or path to image file)
        prompt: Text prompt
        negative_prompt: Negative prompt
        strength: Denoising strength (0.0-1.0)
        seed: Random seed
        sample_steps: Diffusion steps
        cfg_scale: CFG scale
        sample_method: Sampling method
        scheduler: Noise scheduler
        n_threads: Number of threads
        vae_path: Path to VAE
        clip_skip: CLIP skip layers

    Returns:
        List of generated SDImage objects
    """
    # Load image if path provided
    if isinstance(init_image, str):
        init_image = SDImage.load(init_image)

    params = SDContextParams(
        model_path=model_path,
        vae_path=vae_path,
        n_threads=n_threads,
        vae_decode_only=False  # Need encoder for img2img
    )

    with SDContext(params) as ctx:
        return ctx.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=init_image.width,
            height=init_image.height,
            seed=seed,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            sample_method=sample_method,
            scheduler=scheduler,
            init_image=init_image,
            strength=strength,
            clip_skip=clip_skip
        )
