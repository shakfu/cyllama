SHORT=`cd build/llama.cpp && git rev-parse --short HEAD`
TAG=`cd build/llama.cpp && git tag --points-at HEAD`

echo "llama.cpp tag:${TAG} short:${SHORT}"
