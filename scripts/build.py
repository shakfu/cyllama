#!/usr/bin/env python3
"""Build and stage the vendored llama.cpp libraries for xllamacpp."""

from __future__ import annotations

import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "thirdparty" / "llama.cpp"
PREFIX = ROOT / "src" / "llama.cpp"


def log(message: str) -> None:
    print(message, flush=True)


def run(command: list[str], cwd: Path) -> None:
    log("Running: " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def env_is_set(name: str) -> bool:
    return bool(os.environ.get(name))


def split_cmake_args(value: str) -> list[str]:
    if not value:
        return []
    posix = platform.system() != "Windows"
    try:
        parts = shlex.split(value, posix=posix)
    except ValueError as exc:
        raise SystemExit(f"Invalid CMAKE_ARGS: {exc}") from exc
    if not posix:
        parts = [part.strip("\"'") for part in parts]
    return parts


def hip_compiler() -> str:
    try:
        hip_root = subprocess.check_output(
            ["hipconfig", "-l"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("`hipconfig -l` failed while configuring HIP build") from exc
    return str(Path(hip_root) / "clang")


def build_llamacpp() -> None:
    log("update from llama.cpp main repo")
    if not PROJECT.exists():
        raise SystemExit(f"Missing llama.cpp checkout: {PROJECT}")

    system = platform.system()
    machine = platform.machine().lower()
    nproc = os.environ.get("NPROC") or str(os.cpu_count() or 2)

    build_dir = PROJECT / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    cmake_args = [
        "-DBUILD_SHARED_LIBS=OFF",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        "-DCMAKE_INSTALL_LIBDIR=lib",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_LLGUIDANCE=ON",
        "-DLLAMA_BUILD_BORINGSSL=ON",
        "-DLLAMA_OPENSSL=OFF",
    ]
    log("Using BoringSSL (static linking)")

    if env_is_set("XLLAMACPP_RELEASE") and system != "Darwin":
        log("Release mode: disabling native CPU optimizations for portability")
        cmake_args.append("-DGGML_NATIVE=OFF")
    else:
        log("Optimizing for native CPU (GGML_NATIVE=ON by default)")

    if system == "Darwin":
        if not os.environ.get("MACOSX_DEPLOYMENT_TARGET"):
            raise SystemExit("MACOSX_DEPLOYMENT_TARGET must be set for macOS builds")
        cmake_args.append(
            f"-DCMAKE_OSX_DEPLOYMENT_TARGET={os.environ['MACOSX_DEPLOYMENT_TARGET']}"
        )

    user_cmake_args = split_cmake_args(os.environ.get("CMAKE_ARGS", ""))
    cmake_args.extend(user_cmake_args)

    targets = [
        "llama-common-base",
        "llama-common",
        "llama",
        "ggml",
        "ggml-cpu",
        "mtmd",
        "cpp-httplib",
        "server-context",
        "llama-server",
    ]

    if env_is_set("XLLAMACPP_BUILD_CUDA"):
        log("Building for CUDA")
        # CI pipelines pin CUDA_ARCHITECTURES to a curated list to keep build
        # times under the runner limit (a few -real archs + PTX fallbacks).
        #
        # When unset (i.e. a user building locally for their own use), fall back
        # to CMake's "native" keyword. Per the CMake docs this detects the GPUs
        # actually installed on the build machine and compiles SASS only for
        # those architectures. That keeps the build fast and produces fully
        # arch-optimized code for the local hardware -- at the cost of a binary
        # that is not portable to other GPU architectures.
        cuda_archs = os.environ.get("CUDA_ARCHITECTURES") or "native"
        log(f"Using CUDA architectures: {cuda_archs}")
        cmake_args.extend(
            [
                "-DGGML_CUDA=ON",
                "-DGGML_CUDA_FORCE_MMQ=ON",
                f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
            ]
        )
        targets.append("ggml-cuda")
    elif env_is_set("XLLAMACPP_BUILD_HIP"):
        log("Building for AMD GPU")
        cmake_args.extend(
            [
                "-DAMDGPU_TARGETS=gfx1100;gfx1101;gfx1102;gfx1030;gfx1031;gfx1032",
                f"-DCMAKE_HIP_COMPILER={hip_compiler()}",
                "-DGGML_HIP_ROCWMMA_FATTN=ON",
                "-DGGML_HIP=ON",
            ]
        )
        targets.append("ggml-hip")
    elif env_is_set("XLLAMACPP_BUILD_VULKAN"):
        if system == "Darwin":
            cmake_args.append("-DCMAKE_BUILD_RPATH=@loader_path")
            if machine == "x86_64":
                log("Building for Intel with Vulkan")
                cmake_args.extend(["-DGGML_METAL=OFF", "-DGGML_VULKAN=ON"])
                targets.extend(["ggml-blas", "ggml-vulkan"])
            else:
                raise SystemExit(
                    "Building for Apple Silicon with Vulkan is not supported"
                )
        else:
            log("Building with Vulkan")
            cmake_args.append("-DGGML_VULKAN=ON")
            targets.append("ggml-vulkan")
    elif env_is_set("XLLAMACPP_BUILD_AARCH64"):
        log("Building for aarch64")
        cmake_args.append("-DGGML_CPU_ARM_ARCH=armv8-a")
        if "-DGGML_BLAS=ON" in os.environ.get("CMAKE_ARGS", ""):
            log("BLAS is enabled via CMAKE_ARGS, adding ggml-blas to build targets")
            targets.append("ggml-blas")
    elif system == "Darwin":
        cmake_args.append("-DCMAKE_BUILD_RPATH=@loader_path")
        if machine == "x86_64":
            log("Building for Intel")
            cmake_args.append("-DGGML_METAL=OFF")
            targets.append("ggml-blas")
        else:
            log("Building for Apple Silicon")
            cmake_args.append("-DGGML_METAL_EMBED_LIBRARY=ON")
            targets.extend(["ggml-blas", "ggml-metal"])
    else:
        log("Building for non-MacOS CPU")
        if "-DGGML_BLAS=ON" in os.environ.get("CMAKE_ARGS", ""):
            log("BLAS is enabled via CMAKE_ARGS, adding ggml-blas to build targets")
            targets.append("ggml-blas")

    log("Running CMake with arguments: " + " ".join(cmake_args))
    log("Building targets: " + " ".join(targets))

    run(["cmake", "..", *cmake_args], cwd=build_dir)
    run(
        [
            "cmake",
            "--build",
            ".",
            "--config",
            "Release",
            "--parallel",
            nproc,
            "--target",
            *targets,
        ],
        cwd=build_dir,
    )

    shutil.rmtree(PREFIX, ignore_errors=True)
    run([sys.executable, str(ROOT / "scripts" / "copy_libs.py")], cwd=ROOT)


def main() -> int:
    try:
        build_llamacpp()
    except subprocess.CalledProcessError as exc:
        return exc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
