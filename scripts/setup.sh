# scripts/setup.sh [download_last_working] [release-tag]
#
# setup.sh 			: (default run) downloads, builds and install last working release of llama.cpp
# setup.sh 1 		: like default
# setup.sh 0    	: downloads, builds and install bleeding edge llama.cpp from repo
# setup.sh 1 <tag>	: downloads, builds and install <tag> release of llama.cpp

CWD=$(pwd)
THIRDPARTY=${CWD}/thirdparty
LAST_WORKING="b7126"
LLAMACPP_VERSION="${2:-${LAST_WORKING}}"
STABLE_BUILD=0
GET_LAST_WORKING="${1:-$STABLE_BUILD}"

if [ $GET_LAST_WORKING = 1 ]; then
	echo "get last working release: ${LAST_WORKING}"
	BRANCH="--branch ${LLAMACPP_VERSION}"
else
	echo "get bleeding edge llama.cpp from main"
	BRANCH= # bleeding edge (llama.cpp main)
fi


get_llamacpp() {
	echo "update from llama.cpp main repo"
	PREFIX=${THIRDPARTY}/llama.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib

	# Build CMake args based on environment variables
	CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON"

	# Check backend environment variables and add appropriate flags
	if [ "${GGML_METAL:-1}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_METAL=ON"
		echo "✓ Enabling Metal backend"
	fi

	if [ "${GGML_CUDA:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_CUDA=ON"
		echo "✓ Enabling CUDA backend"
	fi

	if [ "${GGML_VULKAN:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_VULKAN=ON"
		echo "✓ Enabling Vulkan backend"
	fi

	if [ "${GGML_SYCL:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_SYCL=ON"
		echo "✓ Enabling SYCL backend"
	fi

	if [ "${GGML_HIP:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_HIP=ON"
		echo "✓ Enabling HIP/ROCm backend"
	fi

	if [ "${GGML_OPENCL:-0}" = "1" ]; then
		CMAKE_ARGS="$CMAKE_ARGS -DGGML_OPENCL=ON"
		echo "✓ Enabling OpenCL backend"
	fi

	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "llama.cpp" ]; then
			git clone ${BRANCH} --recursive https://github.com/ggml-org/llama.cpp.git
		fi && \
		cd llama.cpp && \
		cp common/*.h ${INCLUDE} && \
		cp common/*.hpp ${INCLUDE} && \
		cp ggml/include/*.h ${INCLUDE} && \
		# cp examples/llava/*.h ${INCLUDE} && \
		mkdir -p ${INCLUDE}/nlohmann && \
		cp vendor/nlohmann/*.hpp ${INCLUDE}/nlohmann/ && \
		mkdir -p build && \
		cd build && \
		echo "Building with: cmake .. $CMAKE_ARGS" && \
		cmake .. $CMAKE_ARGS && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp ggml/src/libggml-base.a ${LIB} && \
		cp ggml/src/libggml-cpu.a ${LIB} && \
		if [ "${GGML_METAL:-1}" = "1" ]; then
			cp ggml/src/ggml-blas/libggml-blas.a ${LIB} 2>/dev/null || true && \
			cp ggml/src/ggml-metal/libggml-metal.a ${LIB} 2>/dev/null || true
		fi && \
		if [ "${GGML_CUDA:-0}" = "1" ]; then
			cp ggml/src/ggml-cuda/libggml-cuda.a ${LIB} 2>/dev/null || true
		fi && \
		if [ "${GGML_VULKAN:-0}" = "1" ]; then
			cp ggml/src/ggml-vulkan/libggml-vulkan.a ${LIB} 2>/dev/null || true
		fi && \
		if [ "${GGML_SYCL:-0}" = "1" ]; then
			cp ggml/src/ggml-sycl/libggml-sycl.a ${LIB} 2>/dev/null || true
		fi && \
		if [ "${GGML_HIP:-0}" = "1" ]; then
			cp ggml/src/ggml-hip/libggml-hip.a ${LIB} 2>/dev/null || true
		fi && \
		if [ "${GGML_OPENCL:-0}" = "1" ]; then
			cp ggml/src/ggml-opencl/libggml-opencl.a ${LIB} 2>/dev/null || true
		fi && \
		cp common/libcommon.a ${LIB} && \
		# cp examples/llava/libllava_static.a ${LIB}/libllava.a && \
		cd ${CWD}
}

get_llamacpp_shared() {
	echo "update from llama.cpp main repo"
	PREFIX=${THIRDPARTY}/llama.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "llama.cpp" ]; then
			git clone ${BRANCH} --recursive https://github.com/ggml-org/llama.cpp.git
		fi && \
		cd llama.cpp && \
		cp common/*.h ${INCLUDE} && \
		cp common/*.hpp ${INCLUDE} && \
		cp ggml/include/*.h ${INCLUDE} && \
		# cp examples/llava/*.h ${INCLUDE} && \
		mkdir -p ${INCLUDE}/nlohmann && \
		cp vendor/nlohmann/*.hpp ${INCLUDE}/nlohmann/ && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_NAME_DIR=${LIB} && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp common/libcommon.a ${LIB} && \
		# cp ggml/src/libggml-base.dylib ${LIB} && \
		# cp ggml/src/libggml-cpu.dylib ${LIB} && \
		# cp ggml/src/ggml-blas/libggml-blas.dylib ${LIB} && \
		# cp ggml/src/ggml-metal/libggml-metal.dylib ${LIB} && \
		# cp common/libcommon.dylib ${LIB} && \
		# # cp examples/llava/libllava_static.dylib ${LIB}/libllava.dylib && \
		# mv ${PREFIX}/bin ${CWD}/bin && \
		cd ${CWD}
}

get_whispercpp() {
	echo "update from whisper.cpp main repo"
	PREFIX=${THIRDPARTY}/whisper.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	BIN=${PREFIX}/bin
	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "whisper.cpp" ]; then
			git clone --depth 1 --recursive https://github.com/ggml-org/whisper.cpp.git
		fi && \
		cd whisper.cpp && \
		cp examples/*.h ${INCLUDE} && \
		cp examples/*.hpp ${INCLUDE} && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp examples/libcommon.a ${LIB} && \
		cp -rf bin/* ${BIN} && \
		cd ${CWD}
}

get_stablediffusioncpp() {
	echo "update from stable-diffusion.cpp main repo"
	PREFIX=${THIRDPARTY}/stable-diffusion.cpp
	INCLUDE=${PREFIX}/include
	LIB=${PREFIX}/lib
	BIN=${PREFIX}/bin
	mkdir -p build ${INCLUDE} && \
		cd build && \
		if [ ! -d "stable-diffusion.cpp" ]; then
			git clone --depth 1 --recursive https://github.com/leejet/stable-diffusion.cpp.git
		fi && \
		cd stable-diffusion.cpp && \
		cp *.h ${INCLUDE} && \
		cp *.hpp ${INCLUDE} && \
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp libstable-diffusion.a ${LIB} && \
		# cp -rf bin/* ${BIN} && \
		cd ${CWD}
}

get_llamacpp_python() {
	echo "update from llama-cpp-python main repo"
	PREFIX=${THIRDPARTY}/llama-cpp-python
	mkdir -p build thirdparty && \
		cd build && \
		if [ ! -d "llama-cpp-python" ]; then
			git clone --depth 1 https://github.com/abetlen/llama-cpp-python.git
		fi && \
		rm -rf ${PREFIX} && \
		mkdir -p ${PREFIX} && \
		cp -rf llama-cpp-python/llama_cpp ${PREFIX}/ && \
		rm -rf llama-cpp-python && \
		cd ${CWD}
}


remove_current() {
	echo "remove current"
	rm -rf build thirdparty
}


main() {
	remove_current
	get_llamacpp
	get_whispercpp
	# get_llamacpp_shared
	# get_llamacpp_python
	# get_stablediffusioncpp
}

main
