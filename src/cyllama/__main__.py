"""cyllama CLI: python -m cyllama [command]"""

import platform
import sys


def _parse_system_info(info_str: str) -> dict[str, str]:
    """Parse 'KEY = VALUE | KEY2 = VALUE2 |' format into a dict."""
    result = {}
    for part in info_str.split("|"):
        part = part.strip()
        if "=" in part:
            key, val = part.split("=", 1)
            result[key.strip()] = val.strip()
    return result


def _cpu_features_from_info(info: dict[str, str]) -> list[str]:
    """Extract enabled CPU features from system info dict."""
    cpu_keys = [
        "NEON",
        "AVX",
        "AVX2",
        "AVX512",
        "FMA",
        "ARM_FMA",
        "F16C",
        "FP16_VA",
        "DOTPROD",
        "SSE3",
        "WASM_SIMD",
        "VSX",
    ]
    features = []
    for key in cpu_keys:
        for info_key, val in info.items():
            if info_key.strip().endswith(key) and val == "1":
                features.append(key)
                break
    return features


def _get_build_info() -> dict:
    """Load build info if available."""
    try:
        from . import _build_info

        return {k: v for k, v in vars(_build_info).items() if not k.startswith("_")}
    except ImportError:
        return {}


def cmd_info():
    """Print build and backend information."""
    from . import __version__

    print(f"cyllama {__version__}")
    print(f"Python {platform.python_version()} ({platform.platform()})")
    build_info = _get_build_info()
    print()

    # llama.cpp
    print("llama.cpp:")
    try:
        from .llama import llama_cpp as cy

        cy.llama_backend_init()
        llama_ver = build_info.get("llama_cpp_version", "unknown")
        print(f"  version:       {llama_ver}")
        print(f"  ggml version:  {cy.ggml_version()}")
        print(f"  ggml commit:   {cy.ggml_commit()}")
        print(f"  registries:    {', '.join(cy.ggml_backend_reg_names())}")
        devices = cy.ggml_backend_dev_info()
        if devices:
            print("  devices:")
            for dev in devices:
                print(f"    {dev['name']:20s} [{dev['type']:5s}]  {dev['description']}")
        print(f"  GPU offload:   {cy.llama_supports_gpu_offload()}")
        print(f"  MMAP support:  {cy.llama_supports_mmap()}")
        print(f"  MLOCK support: {cy.llama_supports_mlock()}")
        print(f"  RPC support:   {cy.llama_supports_rpc()}")
    except Exception as e:
        print(f"  not available ({e})")

    print()

    # whisper.cpp
    print("whisper.cpp:")
    try:
        from .whisper import whisper_cpp

        # Load backends so whisper sees GPU registries (mirrors what
        # every upstream whisper.cpp example does in main())
        whisper_cpp.ggml_backend_load_all()

        info_str = whisper_cpp.print_system_info()
        whisper_ver = build_info.get("whisper_cpp_version", "unknown")
        print(f"  version:       {whisper_ver}")
        print(f"  ggml version:  {build_info.get('whisper_cpp_ggml_version', whisper_cpp.version())}")
        info = _parse_system_info(info_str)
        features = _cpu_features_from_info(info)
        # whisper_print_system_info lists dynamically-loaded backends with
        # "BACKEND : feature = val |" format.  Detect backend name prefixes.
        backend_names = ["CUDA", "Metal", "Vulkan", "SYCL", "HIP", "OpenCL"]
        backends = [b for b in backend_names if f"{b} :" in info_str]
        # Also check whisper-specific compile-time backends
        if info.get("COREML") == "1":
            backends.append("CoreML")
        if info.get("OPENVINO") == "1":
            backends.append("OpenVINO")
        print(f"  backends:      {', '.join(backends) if backends else 'CPU'}")
        if features:
            print(f"  CPU features:  {', '.join(features)}")
    except Exception as e:
        print(f"  not available ({e})")

    print()

    # stable-diffusion.cpp
    print("stable-diffusion.cpp:")
    try:
        from .sd import get_system_info, ggml_backend_load_all as sd_load_backends

        # Load backends so sd sees GPU registries
        sd_load_backends()

        sd_ver = build_info.get("stable_diffusion_cpp_version", "unknown")
        print(f"  version:       {sd_ver}")
        print(f"  ggml version:  {build_info.get('stable_diffusion_cpp_ggml_version', 'unknown')}")
        info_str = get_system_info()
        info = _parse_system_info(info_str)
        features = _cpu_features_from_info(info)
        # sd_get_system_info only reports CPU features (old-style ggml_cpu_has_*),
        # so query the ggml backend registry for GPU backends.
        try:
            from .llama import llama_cpp as cy

            regs = cy.ggml_backend_reg_names()
            sd_backends = [r for r in regs if r not in ("CPU", "BLAS")]
            print(f"  backends:      {', '.join(sd_backends) if sd_backends else 'CPU'}")
        except Exception:
            print("  backends:      CPU")
        if features:
            print(f"  CPU features:  {', '.join(features)}")
    except Exception as e:
        print(f"  not available ({e})")


def cmd_version():
    """Print version."""
    from . import __version__

    print(__version__)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python -m cyllama <command>")
        print()
        print("Commands:")
        print("  info      Show build and backend information")
        print("  version   Show version")
        return 0

    cmd = sys.argv[1]
    if cmd == "info":
        cmd_info()
    elif cmd == "version":
        cmd_version()
    else:
        print(f"Unknown command: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
