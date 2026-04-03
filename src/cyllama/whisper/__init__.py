# Ensure platform-specific DLL paths are set before any native extension loads
from ..utils.platform import ensure_native_deps

ensure_native_deps()
