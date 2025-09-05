# scripts/setup.sh [download_last_working] [release-tag]
#
# setup.sh 			: (default run) downloads, builds and install last working release of llama.cpp
# setup.sh 1 		: like default
# setup.sh 0    	: downloads, builds and install bleeding edge llama.cpp from repo
# setup.sh 1 <tag>	: downloads, builds and install <tag> release of llama.cpp

CWD=$(pwd)
THIRDPARTY=${CWD}/thirdparty
LAST_WORKING="b6387"
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
		mkdir -p build && \
		cd build && \
		cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && \
		cmake --build . --config Release && \
		cmake --install . --prefix ${PREFIX} && \
		cp ggml/src/libggml-base.a ${LIB} && \
		cp ggml/src/libggml-cpu.a ${LIB} && \
		cp ggml/src/ggml-blas/libggml-blas.a ${LIB} && \
		cp ggml/src/ggml-metal/libggml-metal.a ${LIB} && \
		cp common/libcommon.a ${LIB} && \
		# cp examples/llava/libllava_static.a ${LIB}/libllava.a && \
		mv ${PREFIX}/bin ${CWD}/bin && \
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
	# get_llamacpp_shared
	# get_llamacpp_python
	# get_whispercpp
	# get_stablediffusioncpp
}

main
