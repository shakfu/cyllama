SHORT=`cd build/llama.cpp && git rev-parse --short HEAD`
TAG=`cd build/llama.cpp && git tag --points-at HEAD`

# echo "tag:${TAG} short:${SHORT}"

git add --all .
git commit -m "synced to llama.cpp tag:${TAG} short:${SHORT}"
git push
