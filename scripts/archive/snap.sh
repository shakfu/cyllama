#!/bin/sh
#
# scripts/snap.sh - Commit and push with version info from all dependencies
#
# Captures git tag/commit info from:
#   - llama.cpp
#   - whisper.cpp
#   - stable-diffusion.cpp
#   - sqlite-vector

set -e

BUILD_DIR="build"

# Helper function to get version info for a repo
get_version_info() {
    local name="$1"
    local dir="$2"

    if [ -d "$dir" ]; then
        local short=$(cd "$dir" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        local tag=$(cd "$dir" && git tag --points-at HEAD 2>/dev/null | head -1)
        if [ -n "$tag" ]; then
            echo "${name}:${tag}"
        else
            echo "${name}:${short}"
        fi
    fi
}

# Collect version info from all dependencies
LLAMA_VER=$(get_version_info "llama.cpp" "${BUILD_DIR}/llama.cpp")
WHISPER_VER=$(get_version_info "whisper.cpp" "${BUILD_DIR}/whisper.cpp")
SD_VER=$(get_version_info "sd.cpp" "${BUILD_DIR}/stable-diffusion.cpp")
VECTOR_VER=$(get_version_info "sqlite-vector" "${BUILD_DIR}/sqlite-vector")

# Build commit message
VERSIONS=""
[ -n "$LLAMA_VER" ] && VERSIONS="${VERSIONS} ${LLAMA_VER}"
[ -n "$WHISPER_VER" ] && VERSIONS="${VERSIONS} ${WHISPER_VER}"
[ -n "$SD_VER" ] && VERSIONS="${VERSIONS} ${SD_VER}"
[ -n "$VECTOR_VER" ] && VERSIONS="${VERSIONS} ${VECTOR_VER}"

# Trim leading space
VERSIONS=$(echo "$VERSIONS" | sed 's/^ //')

if [ -z "$VERSIONS" ]; then
    echo "No dependencies found in ${BUILD_DIR}/"
    exit 1
fi

echo "Syncing with:${VERSIONS}"

git add --all .
git commit -m "synced to ${VERSIONS}"
git push

# Uncomment to also merge to main:
# git switch main
# git merge dev
# git push
# git switch dev
