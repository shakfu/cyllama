#!/bin/bash
# Print CalVer format version: vYYYY.MM.NNNN
# where YYYY.MM is the current year.month, NNNN is the llama.cpp build number
#
# Usage:
#   ./scripts/generate-release-version.sh
#
# Example output: v2026.4.8672

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LLAMACPP_DIR="$PROJECT_DIR/thirdparty/llama.cpp"

# Get current year and month
YEAR=$(date +%Y)
# Strip leading zero from month (PEP 440 does not allow leading zeros)
MONTH=$(date +%-m)

# Get build version number from llama.cpp submodule
if [ ! -d "$LLAMACPP_DIR/.git" ] && [ ! -f "$LLAMACPP_DIR/.git" ]; then
    echo "ERROR: llama.cpp submodule not found at $LLAMACPP_DIR" >&2
    echo "Run 'git submodule update --init --recursive' first." >&2
    exit 1
fi

LLAMACPP_TAG=$(cd "$LLAMACPP_DIR" && git describe --tags --abbrev=0 2>/dev/null)
if [ -z "$LLAMACPP_TAG" ]; then
    echo "ERROR: Could not determine llama.cpp version tag." >&2
    exit 1
fi

# Extract build number (strip 'b' prefix, e.g. b8672 -> 8672)
BUILD_NUM=$(echo "$LLAMACPP_TAG" | sed 's/^b//')
if ! [[ "$BUILD_NUM" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Unexpected llama.cpp tag format: $LLAMACPP_TAG (expected bNNNN)" >&2
    exit 1
fi

TAG="v${YEAR}.${MONTH}.${BUILD_NUM}"

echo "Version info:"
echo "  Date:          ${YEAR}-${MONTH}"
echo "  llama.cpp tag: ${LLAMACPP_TAG}"
echo "  Build number:  ${BUILD_NUM}"
echo "  Release tag:   ${TAG}"
