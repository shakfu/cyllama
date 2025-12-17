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
from libcpp cimport bool as cpp_bool

from .stable_diffusion cimport *

# stb_image declarations
cdef extern from "stb_image.h":
    unsigned char* stbi_load(const char* filename, int* x, int* y, int* channels_in_file, int desired_channels)
    void stbi_image_free(void* retval_from_stbi_load)

cdef extern from "stb_image_write.h":
    int stbi_write_png(const char* filename, int w, int h, int comp, const void* data, int stride_in_bytes)
    int stbi_write_bmp(const char* filename, int w, int h, int comp, const void* data)
    int stbi_write_jpg(const char* filename, int w, int h, int comp, const void* data, int quality)

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
    EPS = EPS_PRED
    V = V_PRED
    EDM_V = EDM_V_PRED
    FLOW = FLOW_PRED
    FLUX_FLOW = FLUX_FLOW_PRED
    FLUX2_FLOW = FLUX2_FLOW_PRED


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
    return sd_get_num_physical_cores()


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

    def save_ppm(self, path: str):
        """
        Save image as PPM (Portable Pixmap) format.

        PPM is a simple uncompressed format that requires no dependencies.
        Most image viewers can open PPM files.

        Args:
            path: Output file path (should end with .ppm)
        """
        if not self.is_valid:
            raise ValueError("Image has no valid data")

        cdef uint32_t w = self._image.width
        cdef uint32_t h = self._image.height
        cdef uint32_t c = self._image.channel
        cdef size_t size = w * h * c
        cdef bytes raw_data

        # Get raw bytes from image data
        raw_data = <bytes>self._image.data[:size]

        with open(path, 'wb') as f:
            # PPM header: P6 for binary RGB
            header = f"P6\n{w} {h}\n255\n".encode('ascii')
            f.write(header)

            if c == 3:
                # RGB - write directly
                f.write(raw_data)
            elif c == 4:
                # RGBA - strip alpha channel
                rgb_data = bytearray(w * h * 3)
                for i in range(w * h):
                    rgb_data[i * 3] = raw_data[i * 4]
                    rgb_data[i * 3 + 1] = raw_data[i * 4 + 1]
                    rgb_data[i * 3 + 2] = raw_data[i * 4 + 2]
                f.write(bytes(rgb_data))
            elif c == 1:
                # Grayscale - expand to RGB
                rgb_data = bytearray(w * h * 3)
                for i in range(w * h):
                    rgb_data[i * 3] = raw_data[i]
                    rgb_data[i * 3 + 1] = raw_data[i]
                    rgb_data[i * 3 + 2] = raw_data[i]
                f.write(bytes(rgb_data))
            else:
                raise ValueError(f"Unsupported channel count: {c}")

    def save_bmp(self, path: str):
        """
        Save image as BMP (Bitmap) format.

        BMP is a simple uncompressed format that requires no dependencies.
        Universally supported by all image viewers and editors.

        Args:
            path: Output file path (should end with .bmp)
        """
        if not self.is_valid:
            raise ValueError("Image has no valid data")

        cdef uint32_t w = self._image.width
        cdef uint32_t h = self._image.height
        cdef uint32_t c = self._image.channel
        cdef size_t size = w * h * c
        cdef bytes raw_data

        # Get raw bytes from image data
        raw_data = <bytes>self._image.data[:size]

        # BMP requires rows to be padded to 4-byte boundaries
        row_size = w * 3
        padding = (4 - (row_size % 4)) % 4
        padded_row_size = row_size + padding
        pixel_data_size = padded_row_size * h

        # BMP file header (14 bytes) + DIB header (40 bytes) = 54 bytes
        file_size = 54 + pixel_data_size

        with open(path, 'wb') as f:
            import struct

            # BMP file header (14 bytes)
            f.write(b'BM')                              # Signature
            f.write(struct.pack('<I', file_size))       # File size
            f.write(struct.pack('<HH', 0, 0))           # Reserved
            f.write(struct.pack('<I', 54))              # Pixel data offset

            # DIB header - BITMAPINFOHEADER (40 bytes)
            f.write(struct.pack('<I', 40))              # Header size
            f.write(struct.pack('<i', w))               # Width
            f.write(struct.pack('<i', h))               # Height (positive = bottom-up)
            f.write(struct.pack('<HH', 1, 24))          # Planes, bits per pixel
            f.write(struct.pack('<I', 0))               # Compression (none)
            f.write(struct.pack('<I', pixel_data_size)) # Image size
            f.write(struct.pack('<i', 2835))            # X pixels per meter (72 DPI)
            f.write(struct.pack('<i', 2835))            # Y pixels per meter (72 DPI)
            f.write(struct.pack('<I', 0))               # Colors in color table
            f.write(struct.pack('<I', 0))               # Important colors

            # Pixel data (bottom-up, BGR order)
            pad_bytes = b'\x00' * padding
            for y in range(h - 1, -1, -1):  # Bottom to top
                for x in range(w):
                    idx = (y * w + x) * c
                    if c >= 3:
                        # BGR order for BMP
                        f.write(bytes([raw_data[idx + 2], raw_data[idx + 1], raw_data[idx]]))
                    elif c == 1:
                        # Grayscale to BGR
                        val = raw_data[idx]
                        f.write(bytes([val, val, val]))
                if padding:
                    f.write(pad_bytes)

    def save_png(self, path: str):
        """
        Save image as PNG format using stb_image_write.

        No external dependencies required. Uses the bundled stb library.

        Args:
            path: Output file path (should end with .png)
        """
        if not self.is_valid:
            raise ValueError("Image has no valid data")

        cdef bytes path_bytes = path.encode('utf-8')
        cdef int result = stbi_write_png(
            path_bytes,
            self._image.width,
            self._image.height,
            self._image.channel,
            self._image.data,
            self._image.width * self._image.channel  # stride
        )
        if result == 0:
            raise IOError(f"Failed to write PNG file: {path}")

    def save_jpg(self, path: str, quality: int = 90):
        """
        Save image as JPEG format using stb_image_write.

        No external dependencies required. Uses the bundled stb library.

        Args:
            path: Output file path (should end with .jpg or .jpeg)
            quality: JPEG quality (1-100, default 90)
        """
        if not self.is_valid:
            raise ValueError("Image has no valid data")

        cdef bytes path_bytes = path.encode('utf-8')
        cdef int result = stbi_write_jpg(
            path_bytes,
            self._image.width,
            self._image.height,
            self._image.channel,
            self._image.data,
            quality
        )
        if result == 0:
            raise IOError(f"Failed to write JPEG file: {path}")

    def save(self, path: str, quality: int = 90):
        """
        Save image to file.

        Uses stb_image_write for PNG, JPEG, and BMP formats (no dependencies).
        Falls back to PIL for other formats if available.

        Args:
            path: Output file path. Format determined by extension.
            quality: JPEG quality (1-100, only used for .jpg/.jpeg)
        """
        ext = path.lower().rsplit('.', 1)[-1] if '.' in path else ''

        # Use stb for common formats (no dependencies)
        if ext == 'png':
            self.save_png(path)
            return
        if ext in ('jpg', 'jpeg'):
            self.save_jpg(path, quality)
            return
        if ext == 'bmp':
            self.save_bmp(path)
            return
        if ext == 'ppm':
            self.save_ppm(path)
            return

        # Try PIL for other formats (gif, tiff, webp, etc.)
        if HAS_PIL:
            img = self.to_pil()
            img.save(path)
        else:
            # Unknown extension without PIL - default to PNG
            if ext in ('gif', 'tiff', 'webp'):
                png_path = path.rsplit('.', 1)[0] + '.png'
                self.save_png(png_path)
                raise ImportError(
                    f"PIL/Pillow required for {ext.upper()} format. "
                    f"Image saved as PNG instead: {png_path}"
                )
            else:
                # Unknown extension - save as PNG
                self.save_png(path)

    @staticmethod
    def from_numpy(arr) -> SDImage:
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
    def from_pil(pil_image) -> SDImage:
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
    def load_ppm(path: str) -> SDImage:
        """
        Load image from PPM (Portable Pixmap) file.

        Supports binary PPM (P6) format. No dependencies required.

        Args:
            path: Input file path

        Returns:
            SDImage: Loaded image
        """
        with open(path, 'rb') as f:
            # Read magic number
            magic = f.readline().strip()
            if magic != b'P6':
                raise ValueError(f"Unsupported PPM format: {magic}. Only P6 (binary RGB) supported.")

            # Skip comments
            line = f.readline()
            while line.startswith(b'#'):
                line = f.readline()

            # Read dimensions
            parts = line.strip().split()
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
            else:
                # Dimensions might be on separate lines
                w = int(parts[0])
                h = int(f.readline().strip())

            # Read max value
            max_val = int(f.readline().strip())
            if max_val != 255:
                raise ValueError(f"Unsupported max value: {max_val}. Only 255 supported.")

            # Read pixel data
            pixel_data = f.read(w * h * 3)

        # Create SDImage
        cdef SDImage img = SDImage()
        img._image.width = w
        img._image.height = h
        img._image.channel = 3

        cdef size_t size = w * h * 3
        img._image.data = <uint8_t*>malloc(size)
        if img._image.data == NULL:
            raise MemoryError("Failed to allocate image data")

        memcpy(img._image.data, <const char*>pixel_data, size)
        img._owns_data = True
        return img

    @staticmethod
    def load_bmp(path: str) -> SDImage:
        """
        Load image from BMP (Bitmap) file.

        Supports uncompressed 24-bit BMP. No dependencies required.

        Args:
            path: Input file path

        Returns:
            SDImage: Loaded image
        """
        import struct

        with open(path, 'rb') as f:
            # BMP file header
            sig = f.read(2)
            if sig != b'BM':
                raise ValueError(f"Not a BMP file: {sig}")

            file_size = struct.unpack('<I', f.read(4))[0]
            f.read(4)  # Reserved
            pixel_offset = struct.unpack('<I', f.read(4))[0]

            # DIB header
            header_size = struct.unpack('<I', f.read(4))[0]
            w = struct.unpack('<i', f.read(4))[0]
            h = struct.unpack('<i', f.read(4))[0]

            # Handle negative height (top-down bitmap)
            top_down = h < 0
            h = abs(h)

            planes = struct.unpack('<H', f.read(2))[0]
            bpp = struct.unpack('<H', f.read(2))[0]
            compression = struct.unpack('<I', f.read(4))[0]

            if bpp != 24:
                raise ValueError(f"Unsupported BMP bit depth: {bpp}. Only 24-bit supported.")
            if compression != 0:
                raise ValueError(f"Compressed BMP not supported.")

            # Seek to pixel data
            f.seek(pixel_offset)

            # Calculate row padding
            row_size = w * 3
            padding = (4 - (row_size % 4)) % 4
            padded_row_size = row_size + padding

            # Read pixel data
            pixel_data = bytearray(w * h * 3)

            for y in range(h):
                row_y = y if top_down else (h - 1 - y)
                row_data = f.read(padded_row_size)
                for x in range(w):
                    idx = (row_y * w + x) * 3
                    src_idx = x * 3
                    # Convert BGR to RGB
                    pixel_data[idx] = row_data[src_idx + 2]
                    pixel_data[idx + 1] = row_data[src_idx + 1]
                    pixel_data[idx + 2] = row_data[src_idx]

        # Create SDImage
        cdef SDImage img = SDImage()
        img._image.width = w
        img._image.height = h
        img._image.channel = 3

        cdef size_t size = w * h * 3
        img._image.data = <uint8_t*>malloc(size)
        if img._image.data == NULL:
            raise MemoryError("Failed to allocate image data")

        cdef bytes pixel_bytes = bytes(pixel_data)
        memcpy(img._image.data, <const char*>pixel_bytes, size)
        img._owns_data = True
        return img

    @staticmethod
    def load(path: str, channels: int = 0) -> SDImage:
        """
        Load image from file using stb_image.

        Supports PNG, JPEG, BMP, TGA, GIF, PSD, HDR, PIC, PNM formats.
        No external dependencies required.

        Args:
            path: Input file path
            channels: Desired number of channels (0=auto, 1=gray, 3=RGB, 4=RGBA)

        Returns:
            SDImage: Loaded image
        """
        # Use built-in PPM loader for PPM files (stb doesn't support PPM well)
        ext = path.lower().rsplit('.', 1)[-1] if '.' in path else ''
        if ext == 'ppm':
            return SDImage.load_ppm(path)

        cdef bytes path_bytes = path.encode('utf-8')
        cdef int w, h, c
        cdef unsigned char* data = stbi_load(path_bytes, &w, &h, &c, channels)

        if data == NULL:
            raise IOError(f"Failed to load image: {path}")

        # Use requested channels if specified, otherwise use file's channels
        cdef int actual_channels = channels if channels > 0 else c

        # Create SDImage and copy data
        cdef SDImage img = SDImage()
        img._image.width = w
        img._image.height = h
        img._image.channel = actual_channels

        cdef size_t size = w * h * actual_channels
        img._image.data = <uint8_t*>malloc(size)
        if img._image.data == NULL:
            stbi_image_free(data)
            raise MemoryError("Failed to allocate image data")

        memcpy(img._image.data, data, size)
        stbi_image_free(data)
        img._owns_data = True
        return img

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
    cdef bytes _llm_path_bytes
    cdef bytes _llm_vision_path_bytes
    cdef bytes _diffusion_model_path_bytes
    cdef bytes _high_noise_diffusion_model_path_bytes
    cdef bytes _vae_path_bytes
    cdef bytes _taesd_path_bytes
    cdef bytes _control_net_path_bytes
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

    # --- Additional model paths ---

    @property
    def clip_vision_path(self) -> Optional[str]:
        """Path to CLIP vision model."""
        if self._params.clip_vision_path:
            return self._params.clip_vision_path.decode('utf-8')
        return None

    @clip_vision_path.setter
    def clip_vision_path(self, value: Optional[str]):
        if value:
            self._clip_vision_path_bytes = value.encode('utf-8')
            self._params.clip_vision_path = self._clip_vision_path_bytes
        else:
            self._params.clip_vision_path = NULL

    @property
    def llm_path(self) -> Optional[str]:
        """Path to LLM text encoder (e.g., Qwen2VL for FLUX2)."""
        if self._params.llm_path:
            return self._params.llm_path.decode('utf-8')
        return None

    @llm_path.setter
    def llm_path(self, value: Optional[str]):
        if value:
            self._llm_path_bytes = value.encode('utf-8')
            self._params.llm_path = self._llm_path_bytes
        else:
            self._params.llm_path = NULL

    @property
    def llm_vision_path(self) -> Optional[str]:
        """Path to LLM vision encoder."""
        if self._params.llm_vision_path:
            return self._params.llm_vision_path.decode('utf-8')
        return None

    @llm_vision_path.setter
    def llm_vision_path(self, value: Optional[str]):
        if value:
            self._llm_vision_path_bytes = value.encode('utf-8')
            self._params.llm_vision_path = self._llm_vision_path_bytes
        else:
            self._params.llm_vision_path = NULL

    @property
    def high_noise_diffusion_model_path(self) -> Optional[str]:
        """Path to high-noise diffusion model (for Wan2.2 MoE)."""
        if self._params.high_noise_diffusion_model_path:
            return self._params.high_noise_diffusion_model_path.decode('utf-8')
        return None

    @high_noise_diffusion_model_path.setter
    def high_noise_diffusion_model_path(self, value: Optional[str]):
        if value:
            self._high_noise_diffusion_model_path_bytes = value.encode('utf-8')
            self._params.high_noise_diffusion_model_path = self._high_noise_diffusion_model_path_bytes
        else:
            self._params.high_noise_diffusion_model_path = NULL

    @property
    def taesd_path(self) -> Optional[str]:
        """Path to TAESD model for fast preview decoding."""
        if self._params.taesd_path:
            return self._params.taesd_path.decode('utf-8')
        return None

    @taesd_path.setter
    def taesd_path(self, value: Optional[str]):
        if value:
            self._taesd_path_bytes = value.encode('utf-8')
            self._params.taesd_path = self._taesd_path_bytes
        else:
            self._params.taesd_path = NULL

    @property
    def control_net_path(self) -> Optional[str]:
        """Path to ControlNet model."""
        if self._params.control_net_path:
            return self._params.control_net_path.decode('utf-8')
        return None

    @control_net_path.setter
    def control_net_path(self, value: Optional[str]):
        if value:
            self._control_net_path_bytes = value.encode('utf-8')
            self._params.control_net_path = self._control_net_path_bytes
        else:
            self._params.control_net_path = NULL

    @property
    def photo_maker_path(self) -> Optional[str]:
        """Path to PhotoMaker model."""
        if self._params.photo_maker_path:
            return self._params.photo_maker_path.decode('utf-8')
        return None

    @photo_maker_path.setter
    def photo_maker_path(self, value: Optional[str]):
        if value:
            self._photo_maker_path_bytes = value.encode('utf-8')
            self._params.photo_maker_path = self._photo_maker_path_bytes
        else:
            self._params.photo_maker_path = NULL

    @property
    def tensor_type_rules(self) -> Optional[str]:
        """Tensor type rules for mixed precision (e.g., '^vae\\.=f16,model\\.=q8_0')."""
        if self._params.tensor_type_rules:
            return self._params.tensor_type_rules.decode('utf-8')
        return None

    @tensor_type_rules.setter
    def tensor_type_rules(self, value: Optional[str]):
        if value:
            self._tensor_type_rules_bytes = value.encode('utf-8')
            self._params.tensor_type_rules = self._tensor_type_rules_bytes
        else:
            self._params.tensor_type_rules = NULL

    # --- Additional boolean/enum parameters ---

    @property
    def sampler_rng_type(self) -> RngType:
        """RNG type for sampler (if different from main RNG)."""
        return RngType(self._params.sampler_rng_type)

    @sampler_rng_type.setter
    def sampler_rng_type(self, value: RngType):
        self._params.sampler_rng_type = <rng_type_t>value

    @property
    def prediction(self) -> Prediction:
        """Prediction type override."""
        return Prediction(self._params.prediction)

    @prediction.setter
    def prediction(self, value: Prediction):
        self._params.prediction = <prediction_t>value

    @property
    def lora_apply_mode(self) -> LoraApplyMode:
        """LoRA application mode."""
        return LoraApplyMode(self._params.lora_apply_mode)

    @lora_apply_mode.setter
    def lora_apply_mode(self, value: LoraApplyMode):
        self._params.lora_apply_mode = <lora_apply_mode_t>value

    @property
    def free_params_immediately(self) -> bool:
        """Free parameters immediately after use."""
        return self._params.free_params_immediately

    @free_params_immediately.setter
    def free_params_immediately(self, value: bool):
        self._params.free_params_immediately = value

    @property
    def keep_clip_on_cpu(self) -> bool:
        """Keep CLIP model on CPU (for low VRAM)."""
        return self._params.keep_clip_on_cpu

    @keep_clip_on_cpu.setter
    def keep_clip_on_cpu(self, value: bool):
        self._params.keep_clip_on_cpu = value

    @property
    def keep_control_net_on_cpu(self) -> bool:
        """Keep ControlNet on CPU (for low VRAM)."""
        return self._params.keep_control_net_on_cpu

    @keep_control_net_on_cpu.setter
    def keep_control_net_on_cpu(self, value: bool):
        self._params.keep_control_net_on_cpu = value

    @property
    def keep_vae_on_cpu(self) -> bool:
        """Keep VAE on CPU (for low VRAM)."""
        return self._params.keep_vae_on_cpu

    @keep_vae_on_cpu.setter
    def keep_vae_on_cpu(self, value: bool):
        self._params.keep_vae_on_cpu = value

    @property
    def tae_preview_only(self) -> bool:
        """Use TAESD only for preview, not final decode."""
        return self._params.tae_preview_only

    @tae_preview_only.setter
    def tae_preview_only(self, value: bool):
        self._params.tae_preview_only = value

    @property
    def diffusion_conv_direct(self) -> bool:
        """Use direct convolution in diffusion model."""
        return self._params.diffusion_conv_direct

    @diffusion_conv_direct.setter
    def diffusion_conv_direct(self, value: bool):
        self._params.diffusion_conv_direct = value

    @property
    def vae_conv_direct(self) -> bool:
        """Use direct convolution in VAE."""
        return self._params.vae_conv_direct

    @vae_conv_direct.setter
    def vae_conv_direct(self, value: bool):
        self._params.vae_conv_direct = value

    @property
    def force_sdxl_vae_conv_scale(self) -> bool:
        """Force conv scale on SDXL VAE."""
        return self._params.force_sdxl_vae_conv_scale

    @force_sdxl_vae_conv_scale.setter
    def force_sdxl_vae_conv_scale(self, value: bool):
        self._params.force_sdxl_vae_conv_scale = value

    @property
    def chroma_use_dit_mask(self) -> bool:
        """Use DiT mask for Chroma models."""
        return self._params.chroma_use_dit_mask

    @chroma_use_dit_mask.setter
    def chroma_use_dit_mask(self, value: bool):
        self._params.chroma_use_dit_mask = value

    @property
    def chroma_use_t5_mask(self) -> bool:
        """Use T5 mask for Chroma models."""
        return self._params.chroma_use_t5_mask

    @chroma_use_t5_mask.setter
    def chroma_use_t5_mask(self, value: bool):
        self._params.chroma_use_t5_mask = value

    @property
    def chroma_t5_mask_pad(self) -> int:
        """T5 mask pad size for Chroma."""
        return self._params.chroma_t5_mask_pad

    @chroma_t5_mask_pad.setter
    def chroma_t5_mask_pad(self, value: int):
        self._params.chroma_t5_mask_pad = value

    @property
    def flow_shift(self) -> float:
        """Flow shift value for SD3.x/Wan models."""
        return self._params.flow_shift

    @flow_shift.setter
    def flow_shift(self, value: float):
        self._params.flow_shift = value

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

    @property
    def shifted_timestep(self) -> int:
        """Shifted timestep for NitroFusion models."""
        return self._params.shifted_timestep

    @shifted_timestep.setter
    def shifted_timestep(self, value: int):
        self._params.shifted_timestep = value

    # --- Guidance parameters ---

    @property
    def img_cfg_scale(self) -> float:
        """Image guidance scale for inpaint/instruct-pix2pix models."""
        return self._params.guidance.img_cfg

    @img_cfg_scale.setter
    def img_cfg_scale(self, value: float):
        self._params.guidance.img_cfg = value

    @property
    def distilled_guidance(self) -> float:
        """Distilled guidance scale for models with guidance input (e.g., FLUX)."""
        return self._params.guidance.distilled_guidance

    @distilled_guidance.setter
    def distilled_guidance(self, value: float):
        self._params.guidance.distilled_guidance = value

    # --- Skip Layer Guidance (SLG) parameters ---

    @property
    def slg_scale(self) -> float:
        """Skip layer guidance scale (for DiT models, 0 = disabled)."""
        return self._params.guidance.slg.scale

    @slg_scale.setter
    def slg_scale(self, value: float):
        self._params.guidance.slg.scale = value

    @property
    def slg_layer_start(self) -> float:
        """SLG enabling point (0.0-1.0)."""
        return self._params.guidance.slg.layer_start

    @slg_layer_start.setter
    def slg_layer_start(self, value: float):
        self._params.guidance.slg.layer_start = value

    @property
    def slg_layer_end(self) -> float:
        """SLG disabling point (0.0-1.0)."""
        return self._params.guidance.slg.layer_end

    @slg_layer_end.setter
    def slg_layer_end(self, value: float):
        self._params.guidance.slg.layer_end = value

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

    @property
    def control_strength(self) -> float:
        """ControlNet strength (0.0-1.0+)."""
        return self._params.control_strength

    @control_strength.setter
    def control_strength(self, value: float):
        self._params.control_strength = value

    # --- VAE Tiling parameters ---

    @property
    def vae_tiling_enabled(self) -> bool:
        """Enable VAE tiling for large images."""
        return self._params.vae_tiling_params.enabled

    @vae_tiling_enabled.setter
    def vae_tiling_enabled(self, value: bool):
        self._params.vae_tiling_params.enabled = value

    @property
    def vae_tile_size(self) -> tuple:
        """VAE tile size (x, y)."""
        return (self._params.vae_tiling_params.tile_size_x,
                self._params.vae_tiling_params.tile_size_y)

    @vae_tile_size.setter
    def vae_tile_size(self, value: tuple):
        self._params.vae_tiling_params.tile_size_x = value[0]
        self._params.vae_tiling_params.tile_size_y = value[1]

    @property
    def vae_tile_overlap(self) -> float:
        """VAE tile overlap ratio (0.0-1.0)."""
        return self._params.vae_tiling_params.target_overlap

    @vae_tile_overlap.setter
    def vae_tile_overlap(self, value: float):
        self._params.vae_tiling_params.target_overlap = value

    # --- EasyCache parameters ---

    @property
    def easycache_enabled(self) -> bool:
        """Enable EasyCache for faster generation."""
        return self._params.easycache.enabled

    @easycache_enabled.setter
    def easycache_enabled(self, value: bool):
        self._params.easycache.enabled = value

    @property
    def easycache_threshold(self) -> float:
        """EasyCache reuse threshold."""
        return self._params.easycache.reuse_threshold

    @easycache_threshold.setter
    def easycache_threshold(self, value: float):
        self._params.easycache.reuse_threshold = value

    @property
    def easycache_range(self) -> tuple:
        """EasyCache start/end percentages."""
        return (self._params.easycache.start_percent,
                self._params.easycache.end_percent)

    @easycache_range.setter
    def easycache_range(self, value: tuple):
        self._params.easycache.start_percent = value[0]
        self._params.easycache.end_percent = value[1]

    # --- Reference image params ---

    @property
    def auto_resize_ref_image(self) -> bool:
        """Auto resize reference images."""
        return self._params.auto_resize_ref_image

    @auto_resize_ref_image.setter
    def auto_resize_ref_image(self, value: bool):
        self._params.auto_resize_ref_image = value

    @property
    def increase_ref_index(self) -> bool:
        """Increase reference index per batch."""
        return self._params.increase_ref_index

    @increase_ref_index.setter
    def increase_ref_index(self, value: bool):
        self._params.increase_ref_index = value

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
                 mask_image: Optional[SDImage] = None,
                 control_image: Optional[SDImage] = None,
                 control_strength: float = 1.0,
                 strength: float = 0.75,
                 clip_skip: int = -1,
                 eta: float = 0.0,
                 slg_scale: float = 0.0,
                 vae_tiling: bool = False) -> List[SDImage]:
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
            mask_image: Mask image for inpainting (white = inpaint)
            control_image: ControlNet control image
            control_strength: ControlNet strength (0.0-1.0+)
            strength: Img2img strength (0.0-1.0)
            clip_skip: Number of CLIP layers to skip
            eta: Eta for DDIM-like samplers
            slg_scale: Skip layer guidance scale (0 = disabled)
            vae_tiling: Enable VAE tiling for large images

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

        # Set advanced parameters
        gen_params.sample_params.eta = eta
        gen_params.sample_params.slg_scale = slg_scale
        gen_params.vae_tiling_enabled = vae_tiling

        # Set init image for img2img
        if init_image is not None:
            gen_params.set_init_image(init_image)

        # Set mask for inpainting
        if mask_image is not None:
            gen_params.set_mask_image(mask_image)

        # Set control image for ControlNet
        if control_image is not None:
            gen_params.set_control_image(control_image, control_strength)

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

    def generate_video(self,
                       prompt: str,
                       negative_prompt: str = "",
                       width: int = 512,
                       height: int = 512,
                       seed: int = -1,
                       video_frames: int = 16,
                       sample_steps: int = 20,
                       cfg_scale: float = 7.0,
                       sample_method: Optional[SampleMethod] = None,
                       scheduler: Optional[Scheduler] = None,
                       init_image: Optional[SDImage] = None,
                       end_image: Optional[SDImage] = None,
                       strength: float = 0.75,
                       clip_skip: int = -1) -> List[SDImage]:
        """
        Generate video frames from a text prompt.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            width: Output frame width
            height: Output frame height
            seed: Random seed (-1 for random)
            video_frames: Number of video frames to generate
            sample_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            sample_method: Sampling method (None for model default)
            scheduler: Noise scheduler (None for model default)
            init_image: Initial image for video (optional)
            end_image: End image for video interpolation (optional)
            strength: Denoising strength (0.0-1.0)
            clip_skip: Number of CLIP layers to skip

        Returns:
            List of SDImage objects representing video frames

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

        # Initialize video generation parameters
        cdef sd_vid_gen_params_t vid_params
        sd_vid_gen_params_init(&vid_params)

        # Set prompt
        cdef bytes prompt_bytes = prompt.encode('utf-8')
        cdef bytes neg_prompt_bytes = negative_prompt.encode('utf-8')
        vid_params.prompt = prompt_bytes
        vid_params.negative_prompt = neg_prompt_bytes

        # Set dimensions and frames
        vid_params.width = width
        vid_params.height = height
        vid_params.video_frames = video_frames
        vid_params.clip_skip = clip_skip
        vid_params.strength = strength
        vid_params.seed = seed

        # Set sample params
        vid_params.sample_params.sample_method = <sample_method_t>sample_method
        vid_params.sample_params.scheduler = <scheduler_t>scheduler
        vid_params.sample_params.sample_steps = sample_steps
        vid_params.sample_params.guidance.txt_cfg = cfg_scale

        # Set init/end images if provided
        if init_image is not None:
            vid_params.init_image = init_image._image
        if end_image is not None:
            vid_params.end_image = end_image._image

        # Generate video
        cdef int num_frames_out = 0
        cdef sd_image_t* result = generate_video(self._ctx, &vid_params, &num_frames_out)

        if result == NULL:
            raise RuntimeError("Video generation failed")

        # Convert results to Python list
        frames = []
        cdef int i
        for i in range(num_frames_out):
            frame = SDImage._from_c_image(result[i], owns_data=True)
            frames.append(frame)

        # Free the array
        free(result)

        return frames


# =============================================================================
# Upscaler class
# =============================================================================

cdef class Upscaler:
    """
    ESRGAN upscaler for image super-resolution.

    Example:
        upscaler = Upscaler("esrgan-x4.bin")
        upscaled = upscaler.upscale(image, factor=4)
    """
    cdef upscaler_ctx_t* _ctx
    cdef bytes _model_path_bytes

    def __cinit__(self):
        self._ctx = NULL

    def __dealloc__(self):
        if self._ctx != NULL:
            free_upscaler_ctx(self._ctx)
            self._ctx = NULL

    def __init__(self,
                 model_path: str,
                 n_threads: int = -1,
                 offload_to_cpu: bool = False,
                 direct: bool = False,
                 tile_size: int = 0):
        """
        Create an upscaler context.

        Args:
            model_path: Path to ESRGAN model file
            n_threads: Number of threads (-1 for auto)
            offload_to_cpu: Offload parameters to CPU
            direct: Use direct convolution
            tile_size: Tile size for processing (0 for default)
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._model_path_bytes = model_path.encode('utf-8')

        if n_threads < 0:
            n_threads = sd_get_num_physical_cores()

        self._ctx = new_upscaler_ctx(
            self._model_path_bytes,
            offload_to_cpu,
            direct,
            n_threads,
            tile_size
        )

        if self._ctx == NULL:
            raise RuntimeError(f"Failed to load upscaler model: {model_path}")

    @property
    def is_valid(self) -> bool:
        """Check if upscaler is valid."""
        return self._ctx != NULL

    @property
    def upscale_factor(self) -> int:
        """Get the upscale factor for this model."""
        if self._ctx == NULL:
            return 0
        return get_upscale_factor(self._ctx)

    def upscale(self, image: SDImage, factor: int = 0) -> SDImage:
        """
        Upscale an image.

        Args:
            image: Input image to upscale
            factor: Upscale factor (0 to use model's default)

        Returns:
            Upscaled SDImage

        Raises:
            RuntimeError: If upscaling fails
        """
        if self._ctx == NULL:
            raise RuntimeError("Upscaler not initialized")

        if factor == 0:
            factor = self.upscale_factor

        cdef sd_image_t result = upscale(self._ctx, image._image, factor)

        if result.data == NULL:
            raise RuntimeError("Upscaling failed")

        return SDImage._from_c_image(result, owns_data=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self._ctx != NULL:
            free_upscaler_ctx(self._ctx)
            self._ctx = NULL


# =============================================================================
# Model conversion function
# =============================================================================

def convert_model(
    input_path: str,
    output_path: str,
    output_type: SDType = SDType.F16,
    vae_path: Optional[str] = None,
    tensor_type_rules: Optional[str] = None
) -> bool:
    """
    Convert a model to a different format/quantization.

    Args:
        input_path: Path to input model
        output_path: Path for output model
        output_type: Output quantization type
        vae_path: Path to VAE model (optional)
        tensor_type_rules: Custom tensor type rules (optional)

    Returns:
        True if conversion successful

    Raises:
        FileNotFoundError: If input model not found
        RuntimeError: If conversion fails
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input model not found: {input_path}")

    cdef bytes input_bytes = input_path.encode('utf-8')
    cdef bytes output_bytes = output_path.encode('utf-8')
    cdef bytes vae_bytes
    cdef bytes rules_bytes
    cdef const char* vae_ptr = NULL
    cdef const char* rules_ptr = NULL

    if vae_path:
        vae_bytes = vae_path.encode('utf-8')
        vae_ptr = vae_bytes

    if tensor_type_rules:
        rules_bytes = tensor_type_rules.encode('utf-8')
        rules_ptr = rules_bytes

    cdef bint success = convert(
        input_bytes,
        vae_ptr,
        output_bytes,
        <sd_type_t>output_type,
        rules_ptr
    )

    if not success:
        raise RuntimeError(f"Model conversion failed: {input_path} -> {output_path}")

    return True


# =============================================================================
# Preprocessing functions
# =============================================================================

def canny_preprocess(
    image: SDImage,
    high_threshold: float = 0.8,
    low_threshold: float = 0.1,
    weak: float = 0.5,
    strong: float = 1.0,
    inverse: bool = False
) -> bool:
    """
    Apply Canny edge detection preprocessing to an image.

    This is useful for ControlNet conditioning.

    Args:
        image: Input image (modified in-place)
        high_threshold: High threshold for edge detection
        low_threshold: Low threshold for edge detection
        weak: Weak edge value
        strong: Strong edge value
        inverse: Invert the result

    Returns:
        True if preprocessing successful
    """
    cdef bint success = preprocess_canny(
        image._image,
        high_threshold,
        low_threshold,
        weak,
        strong,
        inverse
    )
    return success


# =============================================================================
# Preview callback support
# =============================================================================

cdef object _preview_callback = None

cdef void _preview_callback_wrapper(int step, int frame_count, sd_image_t* frames, cpp_bool is_noisy, void* data) noexcept with gil:
    """Internal wrapper for preview callback."""
    global _preview_callback
    cdef int i
    if _preview_callback is not None:
        try:
            # Convert frames to Python list
            frame_list = []
            for i in range(frame_count):
                # Create SDImage without owning data (preview only)
                img = SDImage._from_c_image(frames[i], owns_data=False)
                frame_list.append(img)
            _preview_callback(step, frame_list, bool(is_noisy))
        except Exception:
            pass  # Ignore exceptions in callback


def set_preview_callback(
    callback: Optional[Callable[[int, List[SDImage], bool], None]],
    mode: PreviewMode = PreviewMode.NONE,
    interval: int = 1,
    denoised: bool = True,
    noisy: bool = False
) -> None:
    """
    Set a callback for generation previews.

    The callback receives:
    - step: Current step number
    - frames: List of preview images
    - is_noisy: Whether the preview is noisy (not denoised)

    Args:
        callback: Callback function or None to disable
        mode: Preview mode (NONE, PROJ, TAE, VAE)
        interval: How often to call the callback (every N steps)
        denoised: Include denoised previews
        noisy: Include noisy previews
    """
    global _preview_callback
    _preview_callback = callback

    if callback is None:
        sd_set_preview_callback(NULL, PREVIEW_NONE, 1, False, False, NULL)
    else:
        sd_set_preview_callback(
            _preview_callback_wrapper,
            <preview_t>mode,
            interval,
            denoised,
            noisy,
            NULL
        )


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
    taesd_path: Optional[str] = None,
    clip_l_path: Optional[str] = None,
    clip_g_path: Optional[str] = None,
    t5xxl_path: Optional[str] = None,
    control_net_path: Optional[str] = None,
    clip_skip: int = -1,
    eta: float = 0.0,
    slg_scale: float = 0.0,
    vae_tiling: bool = False,
    offload_to_cpu: bool = False,
    keep_clip_on_cpu: bool = False,
    keep_vae_on_cpu: bool = False,
    diffusion_flash_attn: bool = False
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
        taesd_path: Path to TAESD for fast previews
        clip_l_path: Path to CLIP-L (for SDXL/SD3)
        clip_g_path: Path to CLIP-G (for SDXL/SD3)
        t5xxl_path: Path to T5-XXL (for SD3/FLUX)
        control_net_path: Path to ControlNet model
        clip_skip: CLIP skip layers
        eta: Eta for DDIM-like samplers
        slg_scale: Skip layer guidance scale (0 = disabled)
        vae_tiling: Enable VAE tiling for large images
        offload_to_cpu: Offload model to CPU (low VRAM)
        keep_clip_on_cpu: Keep CLIP on CPU
        keep_vae_on_cpu: Keep VAE on CPU
        diffusion_flash_attn: Use flash attention

    Returns:
        List of generated SDImage objects
    """
    params = SDContextParams(
        model_path=model_path,
        vae_path=vae_path,
        clip_l_path=clip_l_path,
        clip_g_path=clip_g_path,
        t5xxl_path=t5xxl_path,
        n_threads=n_threads
    )

    # Set additional context params
    if taesd_path:
        params.taesd_path = taesd_path
    if control_net_path:
        params.control_net_path = control_net_path
    params.offload_params_to_cpu = offload_to_cpu
    params.keep_clip_on_cpu = keep_clip_on_cpu
    params.keep_vae_on_cpu = keep_vae_on_cpu
    params.diffusion_flash_attn = diffusion_flash_attn

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
            eta=eta,
            slg_scale=slg_scale,
            vae_tiling=vae_tiling,
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
