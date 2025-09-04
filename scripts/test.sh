LAST_WORKING="b6374"
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

echo "LAST_WORKING = ${LAST_WORKING}"
echo "LLAMACPP_VERSION = ${LLAMACPP_VERSION}"

echo "STABLE_BUILD = ${STABLE_BUILD}"
echo "GET_LAST_WORKING = ${GET_LAST_WORKING}"

