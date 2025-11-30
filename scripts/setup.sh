#!/bin/sh

# scripts/setup.sh [download_last_working] [release-tag]
#
# setup.sh 			: (default run) downloads, builds and install last working release of llama.cpp
# setup.sh 1 		: like default
# setup.sh 0    	: downloads, builds and install bleeding edge llama.cpp from repo
# setup.sh 1 <tag>	: downloads, builds and install <tag> release of llama.cpp

CWD=$(pwd)
THIRDPARTY=${CWD}/thirdparty
LAST_WORKING_LLAMACPP="b7126"
LAST_WORKING_SDCPP="master-377-2034588"
LAST_WORKING_WHISPERCPP="v1.8.2"
STABLE_BUILD=0
GET_LAST_WORKING_LLAMACPP="${1:-$STABLE_BUILD}"

# Detect OS
OS_TYPE=$(uname -s)
IS_WINDOWS=0
IS_MACOS=0
IS_LINUX=0

case "$OS_TYPE" in
	Darwin)
		IS_MACOS=1
		METAL_DEFAULT=1
		LIB_PREFIX="lib"
		STATIC_LIB_EXT=".a"
		CMAKE_BUILD_SUBDIR=""
		;;
	MINGW*|MSYS*|CYGWIN*)
		IS_WINDOWS=1
		METAL_DEFAULT=0
		LIB_PREFIX=""
		STATIC_LIB_EXT=".lib"
		CMAKE_BUILD_SUBDIR="/Release"
		;;
	Linux)
		IS_LINUX=1
		METAL_DEFAULT=0
		LIB_PREFIX="lib"
		STATIC_LIB_EXT=".a"
		CMAKE_BUILD_SUBDIR=""
		;;
	*)
		echo "Unknown OS: $OS_TYPE"
		METAL_DEFAULT=0
		LIB_PREFIX="lib"
		STATIC_LIB_EXT=".a"
		CMAKE_BUILD_SUBDIR=""
		;;
esac

echo "Detected OS: $OS_TYPE"
echo "  IS_WINDOWS: $IS_WINDOWS"
echo "  IS_MACOS: $IS_MACOS"
echo "  IS_LINUX: $IS_LINUX"
echo "  Library prefix: '$LIB_PREFIX'"
echo "  Static lib extension: '$STATIC_LIB_EXT'"

if [ $GET_LAST_WORKING_LLAMACPP -eq 1 ]; then
	echo "get last working release: ${LAST_WORKING_LLAMACPP}"
	BRANCH="--branch ${LAST_WORKING_LLAMACPP}"
else
	echo "get bleeding edge llama.cpp from main"
	BRANCH= # bleeding edge (llama.cpp main)
fi

# Helper function to copy library with platform-specific naming
# Usage: copy_lib <source_dir> <lib_name> <dest_dir>
copy_lib() {
	SRC_DIR="$1"
	LIB_NAME="$2"
	DEST_DIR="$3"

	# Try Release subdirectory first (Windows), then direct path (Unix)
	if [ -f "${SRC_DIR}/Release/${LIB_NAME}${STATIC_LIB_EXT}" ]; then
		cp "${SRC_DIR}/Release/${LIB_NAME}${STATIC_LIB_EXT}" "${DEST_DIR}/" && \
		echo "  Copied ${LIB_NAME}${STATIC_LIB_EXT} from Release/"
	elif [ -f "${SRC_DIR}/${LIB_PREFIX}${LIB_NAME}${STATIC_LIB_EXT}" ]; then
		cp "${SRC_DIR}/${LIB_PREFIX}${LIB_NAME}${STATIC_LIB_EXT}" "${DEST_DIR}/" && \
		echo "  Copied ${LIB_PREFIX}${LIB_NAME}${STATIC_LIB_EXT}"
	elif [ -f "${SRC_DIR}/${LIB_NAME}${STATIC_LIB_EXT}" ]; then
		cp "${SRC_DIR}/${LIB_NAME}${STATIC_LIB_EXT}" "${DEST_DIR}/" && \
		echo "  Copied ${LIB_NAME}${STATIC_LIB_EXT}"
	else
		echo "  Warning: ${LIB_NAME} not found in ${SRC_DIR}"
		return 1
	fi
	return 0
}

get_llamacpp() {
	echo ""
	echo "=== Building llama.cpp ==="
	PREFIX=${THIRDPARTY}/llama.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib

	# Build CMake args based on environment variables
	CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DLLAMA_CURL=OFF"

	# Explicitly disable Metal on non-macOS
	if [ "$IS_MACOS" = "0" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=OFF"
	fi

	# Check backend environment variables and add appropriate flags
	if [ "${GGML_METAL:-$METAL_DEFAULT}" = "1" ] && [ "$IS_MACOS" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON"
		echo "Enabling Metal backend"
	fi

	if [ "${GGML_CUDA:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
		echo "Enabling CUDA backend"
	fi

	if [ "${GGML_VULKAN:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_VULKAN=ON"
		echo "Enabling Vulkan backend"
	fi

	if [ "${GGML_SYCL:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_SYCL=ON"
		echo "Enabling SYCL backend"
	fi

	if [ "${GGML_HIP:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"
		echo "Enabling HIP/ROCm backend"
	fi

	if [ "${GGML_OPENCL:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_OPENCL=ON"
		echo "Enabling OpenCL backend"
	fi

	mkdir -p build ${INCLUDE} ${LIB} && \
		cd build && \
		if [ ! -d "llama.cpp" ]; then
			git clone ${BRANCH} --depth=1 --recursive --shallow-submodules https://github.com/ggml-org/llama.cpp.git
		fi && \
		cd llama.cpp && \
		cp common/*.h ${INCLUDE} && \
		cp common/*.hpp ${INCLUDE} && \
		cp ggml/include/*.h ${INCLUDE} && \
		mkdir -p ${INCLUDE}/nlohmann && \
		cp vendor/nlohmann/*.hpp ${INCLUDE}/nlohmann/ && \
		mkdir -p build && \
		cd build && \
		echo "Building with: cmake .. $CMAKE_ARGS" && \
		cmake .. $CMAKE_ARGS && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		echo "Copying additional libraries..." && \
		copy_lib "ggml/src" "ggml-base" "${LIB}" && \
		copy_lib "ggml/src" "ggml-cpu" "${LIB}" && \
		copy_lib "common" "common" "${LIB}"

	# Copy backend-specific libraries
	if [ "${GGML_METAL:-$METAL_DEFAULT}" = "1" ] && [ "$IS_MACOS" = "1" ]; then
		copy_lib "ggml/src/ggml-blas" "ggml-blas" "${LIB}" 2>/dev/null || true
		copy_lib "ggml/src/ggml-metal" "ggml-metal" "${LIB}" 2>/dev/null || true
	fi

	if [ "${GGML_CUDA:-0}" = "1" ]; then
		copy_lib "ggml/src/ggml-cuda" "ggml-cuda" "${LIB}" 2>/dev/null || true
	fi

	if [ "${GGML_VULKAN:-0}" = "1" ]; then
		copy_lib "ggml/src/ggml-vulkan" "ggml-vulkan" "${LIB}" 2>/dev/null || true
	fi

	if [ "${GGML_SYCL:-0}" = "1" ]; then
		copy_lib "ggml/src/ggml-sycl" "ggml-sycl" "${LIB}" 2>/dev/null || true
	fi

	if [ "${GGML_HIP:-0}" = "1" ]; then
		copy_lib "ggml/src/ggml-hip" "ggml-hip" "${LIB}" 2>/dev/null || true
	fi

	if [ "${GGML_OPENCL:-0}" = "1" ]; then
		copy_lib "ggml/src/ggml-opencl" "ggml-opencl" "${LIB}" 2>/dev/null || true
	fi

	cd ${CWD} || exit
	echo "llama.cpp build complete!"
}

get_llamacpp_shared() {
	echo "update from llama.cpp main repo"
	PREFIX=${THIRDPARTY}/llama.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "llama.cpp" ]; then
			git clone ${BRANCH} --depth=1 --recursive --shallow-submodules https://github.com/ggml-org/llama.cpp.git
		fi && \
		cd llama.cpp && \
		cp common/*.h ${INCLUDE} && \
		cp common/*.hpp ${INCLUDE} && \
		cp ggml/include/*.h ${INCLUDE} && \
		mkdir -p ${INCLUDE}/nlohmann && \
		cp vendor/nlohmann/*.hpp ${INCLUDE}/nlohmann/ && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_NAME_DIR=${LIB} && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		copy_lib "common" "common" "${LIB}" && \
		cd ${CWD} || exit
}

get_whispercpp() {
	echo ""
	echo "=== Building whisper.cpp ==="
	WHISPERCPP_VERSION=${LAST_WORKING_WHISPERCPP}
	PREFIX=${THIRDPARTY}/whisper.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	BIN=${PREFIX}/bin
	mkdir -p build ${INCLUDE} ${LIB} ${BIN} && \
		cd build && \
		if [ ! -d "whisper.cpp" ]; then
			git clone --branch ${WHISPERCPP_VERSION} --depth=1 --recursive --shallow-submodules https://github.com/ggml-org/whisper.cpp.git
		fi && \
		cd whisper.cpp && \
		cp examples/*.h ${INCLUDE} 2>/dev/null || true && \
		cp examples/*.hpp ${INCLUDE} 2>/dev/null || true && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		echo "Copying additional libraries..." && \
		copy_lib "examples" "common" "${LIB}" && \
		cp -rf bin${CMAKE_BUILD_SUBDIR}/* ${BIN} 2>/dev/null || true && \
		cd ${CWD} || exit
	echo "whisper.cpp build complete!"
}

get_stablediffusioncpp() {
	echo ""
	echo "=== Building stable-diffusion.cpp ==="
	SDCPP_VERSION=${LAST_WORKING_SDCPP}
	PREFIX=${THIRDPARTY}/stable-diffusion.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	BIN=${PREFIX}/bin
	mkdir -p build ${INCLUDE} ${LIB} ${BIN} && \
		cd build && \
		if [ ! -d "stable-diffusion.cpp" ]; then
			git clone --branch ${SDCPP_VERSION} --depth=1 --recursive --shallow-submodules https://github.com/leejet/stable-diffusion.cpp.git
		fi && \
		cd stable-diffusion.cpp && \
		cp *.h ${INCLUDE} 2>/dev/null || true && \
		cp *.hpp ${INCLUDE} 2>/dev/null || true && \
		cp thirdparty/stb_image.h ${INCLUDE} && \
		cp thirdparty/stb_image_write.h ${INCLUDE} && \
		cp thirdparty/stb_image_resize.h ${INCLUDE} && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		echo "Copying additional libraries..." && \
		copy_lib "." "stable-diffusion" "${LIB}" && \
		cd ${CWD} || exit
	echo "stable-diffusion.cpp build complete!"
}

remove_current() {
	echo "Cleaning previous builds..."
	rm -rf build thirdparty
}


main() {
	remove_current
	get_llamacpp
	get_whispercpp
	get_stablediffusioncpp
	# get_llamacpp_shared
	echo ""
	echo "=== Setup Complete ==="
	echo "Libraries installed to: ${THIRDPARTY}"
}

main
