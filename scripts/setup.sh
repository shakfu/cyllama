# scripts/setup.sh [download_last_working] [release-tag]
#
# setup.sh 			: (default run) downloads, builds and install last working release of llama.cpp
# setup.sh 1 		: like default
# setup.sh 0    	: downloads, builds and install bleeding edge llama.cpp from repo
# setup.sh 1 <tag>	: downloads, builds and install <tag> release of llama.cpp

CWD=$(pwd)
THIRDPARTY=${CWD}/thirdparty

build_llamacpp() {
	echo "update from llama.cpp main repo"
	PROJECT=${THIRDPARTY}/llama.cpp
	PREFIX=${CWD}/src/llama.cpp
	NPROC=${NPROC:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)}
	cd ${PROJECT} && \
		mkdir -p build &&
		cd build
  # Base CMake arguments
  local cmake_args=(
    "-DBUILD_SHARED_LIBS=OFF"
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
    "-DCMAKE_INSTALL_LIBDIR=lib"
    "-DLLAMA_CURL=OFF"
    "-DLLAMA_LLGUIDANCE=ON"
  )

  # Use BoringSSL (static linking) on all platforms for maximum portability
  echo "Using BoringSSL (static linking)"
  cmake_args+=("-DLLAMA_BUILD_BORINGSSL=ON")
  cmake_args+=("-DLLAMA_OPENSSL=OFF")

  # Release builds on non-macOS: disable native CPU optimizations for portability
  # macOS: always enable native optimizations (same CPU per architecture)
  # User self-builds: enable native CPU optimizations for best performance (CMake default)
  if [[ -n "${XLLAMACPP_RELEASE:-}" && "$(uname -s)" != "Darwin" ]]; then
    echo "Release mode: disabling native CPU optimizations for portability"
    cmake_args+=("-DGGML_NATIVE=OFF")
  else
    echo "Optimizing for native CPU (GGML_NATIVE=ON by default)"
  fi

  # Add any additional CMake arguments from environment
  if [ -n "${CMAKE_ARGS}" ]; then
    cmake_args+=(${CMAKE_ARGS})
  fi

  # Build targets
  local targets=("llama-common-base" "llama-common" "llama" "ggml" "ggml-cpu" "mtmd" "cpp-httplib" "server-context" "llama-server")
  
  if [[ -n "${XLLAMACPP_BUILD_CUDA}" ]]; then
    echo "Building for CUDA"

    # Get CUDA architectures from nvcc if CUDA_ARCHITECTURES not set
    local cuda_archs="${CUDA_ARCHITECTURES:-}"
    if [[ -z "${cuda_archs}" ]]; then
      echo "=== Detecting supported GPU architectures ==="
      nvcc --list-gpu-arch
      # Filter for compute capability >= 70, replace 120 with 120a for Blackwell optimizations
	  # GPU		arch
      # V100	70
      # T4		75
      # A100	80
      # 3090	86
      # 4090	89
      cuda_archs=$(nvcc --list-gpu-arch | grep -E '^(sm|compute)_[0-9]+$' | sed -E 's/(sm|compute)_//' | sort -u | awk '$1 >= 70' | tr '\n' ';' | sed 's/;$//' | sed 's/\b120\b/120a/g')
    fi
    echo "Using CUDA architectures: ${cuda_archs}"

    cmake_args+=(
      "-DGGML_CUDA=ON"
      "-DGGML_CUDA_FORCE_MMQ=ON"
	  "-DCMAKE_CUDA_ARCHITECTURES=${cuda_archs}"
    )
    targets+=("ggml-cuda")
  elif [[ -n "${XLLAMACPP_BUILD_HIP}" ]]; then
    echo "Building for AMD GPU"
    cmake_args+=(
      "-DAMDGPU_TARGETS=gfx1100;gfx1101;gfx1102;gfx1030;gfx1031;gfx1032"
      "-DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang"
      "-DGGML_HIP_ROCWMMA_FATTN=ON"
      "-DGGML_HIP=ON"
    )
    targets+=("ggml-hip")
  elif [[ -n "${XLLAMACPP_BUILD_VULKAN}" ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      cmake_args+=("-DCMAKE_BUILD_RPATH=@loader_path")
      if [[ "$(uname -m)" == "x86_64" ]]; then
        echo "Building for Intel with Vulkan"
        cmake_args+=(
          "-DGGML_METAL=OFF"
          "-DGGML_VULKAN=ON"
        )
        targets+=("ggml-blas" "ggml-vulkan")
      else
        echo "Building for Apple Silicon with Vulkan is not supported"
        exit 1
      fi
    else
      echo "Building with Vulkan"
      cmake_args+=(
        "-DGGML_VULKAN=ON"
      )
      targets+=("ggml-vulkan")
    fi
  elif [[ -n "${XLLAMACPP_BUILD_AARCH64}" ]]; then
    echo "Building for aarch64"
    cmake_args+=(
      "-DGGML_CPU_ARM_ARCH=armv8-a"
    )
    # Add ggml-blas target if BLAS is enabled via CMAKE_ARGS
    if [[ "${CMAKE_ARGS:-}" == *"-DGGML_BLAS=ON"* ]]; then
      echo "BLAS is enabled via CMAKE_ARGS, adding ggml-blas to build targets"
      targets+=("ggml-blas")
    fi
  else
    if [[ "$(uname -s)" == "Darwin" ]]; then
      cmake_args+=("-DCMAKE_BUILD_RPATH=@loader_path")
      if [[ "$(uname -m)" == "x86_64" ]]; then
        echo "Building for Intel"
        cmake_args+=("-DGGML_METAL=OFF")
        targets+=("ggml-blas")
      else
        echo "Building for Apple Silicon"
        cmake_args+=("-DGGML_METAL_EMBED_LIBRARY=ON")
        targets+=("ggml-blas" "ggml-metal")
      fi
    else
      echo "Building for non-MacOS CPU"
      # Let CMake handle GGML_BLAS from environment
      if [[ "${CMAKE_ARGS:-}" == *"-DGGML_BLAS=ON"* ]]; then
        echo "BLAS is enabled via CMAKE_ARGS, adding ggml-blas to build targets"
        targets+=("ggml-blas")
      fi
    fi
  fi

  # Run CMake and build
  echo "Running CMake with arguments: ${cmake_args[*]}"
  echo "Building targets: ${targets[*]}"

  cmake .. "${cmake_args[@]}"
  if [ $? -ne 0 ]; then
    echo "ERROR: CMake configure failed. Aborting build."
    exit 1
  fi

  cmake --build . --config Release -j ${NPROC} --target "${targets[@]}"
  if [ $? -ne 0 ]; then
    echo "ERROR: CMake build failed. Aborting build."
    exit 1
  fi

  rm -rf ${PREFIX}
  python ${CWD}/scripts/copy_libs.py
  cd ${CWD}
}

build_llamacpp
