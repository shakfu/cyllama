# Ensure platform-specific DLL paths are set before any native extension loads
from ..utils.platform import ensure_native_deps

ensure_native_deps()

from .whisper_cpp import (
    WHISPER,
    WhisperSamplingStrategy,
    WhisperAheadsPreset,
    WhisperGretype,
    WhisperContextParams,
    WhisperVadParams,
    WhisperFullParams,
    WhisperTokenData,
    WhisperContext,
    WhisperState,
    ggml_backend_load_all,
    disable_logging,
    version,
    print_system_info,
    lang_max_id,
    lang_id,
    lang_str,
    lang_str_full,
)

__all__ = [
    "WHISPER",
    "WhisperSamplingStrategy",
    "WhisperAheadsPreset",
    "WhisperGretype",
    "WhisperContextParams",
    "WhisperVadParams",
    "WhisperFullParams",
    "WhisperTokenData",
    "WhisperContext",
    "WhisperState",
    "ggml_backend_load_all",
    "disable_logging",
    "version",
    "print_system_info",
    "lang_max_id",
    "lang_id",
    "lang_str",
    "lang_str_full",
]
