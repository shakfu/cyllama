#!/usr/bin/env bash
# Re-vendor jinja2 and markupsafe into src/cyllama/_vendor.
#
# Usage: ./scripts/vendor_jinja2.sh [JINJA2_VERSION] [MARKUPSAFE_VERSION]
#
# Defaults to the latest stable releases. After running, verify the
# tests pass and bump the version table in src/cyllama/_vendor/README.md.
#
# This script:
#   1. Downloads jinja2 and markupsafe wheels via `pip download`.
#   2. Extracts the .py source files into src/cyllama/_vendor/.
#   3. Removes markupsafe's optional _speedups.c C extension and any
#      compiled .so artifacts (we use the pure-Python _native.py path).
#   4. Rewrites all `from markupsafe import` / `import markupsafe`
#      lines inside the vendored jinja2 to reference
#      `cyllama._vendor.markupsafe` instead. Same pattern as
#      pip._vendor and setuptools._vendor.
#   5. Copies the LICENSE.txt files to LICENSE.jinja2 / LICENSE.markupsafe
#      so the vendor directory is self-documenting for license compliance.
#
# The rewriting is idempotent: running this script multiple times with
# the same versions produces the same output.

set -euo pipefail

JINJA2_VERSION="${1:-3.1.6}"
MARKUPSAFE_VERSION="${2:-3.0.3}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR_DIR="${REPO_ROOT}/src/cyllama/_vendor"

if [[ ! -d "${VENDOR_DIR}" ]]; then
    echo "ERROR: vendor directory does not exist: ${VENDOR_DIR}" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "Downloading jinja2==${JINJA2_VERSION} and markupsafe==${MARKUPSAFE_VERSION}..."
python -m pip download \
    --no-deps --no-binary=:none: --dest "${WORK_DIR}" \
    "jinja2==${JINJA2_VERSION}" "markupsafe==${MARKUPSAFE_VERSION}"

echo "Extracting wheels..."
for whl in "${WORK_DIR}"/*.whl; do
    python -m zipfile -e "${whl}" "${WORK_DIR}/extracted"
done

echo "Removing old vendored jinja2 and markupsafe..."
rm -rf "${VENDOR_DIR}/jinja2" "${VENDOR_DIR}/markupsafe"

echo "Copying new sources..."
cp -r "${WORK_DIR}/extracted/jinja2" "${VENDOR_DIR}/jinja2"
cp -r "${WORK_DIR}/extracted/markupsafe" "${VENDOR_DIR}/markupsafe"

echo "Stripping markupsafe C extension (we use the pure-Python _native fallback)..."
rm -f "${VENDOR_DIR}/markupsafe/_speedups.c" \
      "${VENDOR_DIR}/markupsafe/_speedups.pyi" \
      "${VENDOR_DIR}/markupsafe"/_speedups.cpython-*.so \
      "${VENDOR_DIR}/markupsafe"/_speedups*.pyd

echo "Rewriting markupsafe imports inside vendored jinja2..."
# Files known to import markupsafe (verified against jinja2 3.1.6).
# Re-grepping on every release would catch any new files.
JINJA_FILES_WITH_MARKUPSAFE_IMPORTS=$(
    cd "${VENDOR_DIR}/jinja2" && \
    grep -lE '^from markupsafe import|^import markupsafe$' *.py || true
)
for f in ${JINJA_FILES_WITH_MARKUPSAFE_IMPORTS}; do
    echo "  rewriting ${f}"
    sed -i.bak \
        -e 's/^from markupsafe import/from cyllama._vendor.markupsafe import/' \
        -e 's/^import markupsafe$/import cyllama._vendor.markupsafe as markupsafe/' \
        "${VENDOR_DIR}/jinja2/${f}"
    rm -f "${VENDOR_DIR}/jinja2/${f}.bak"
done

echo "Rewriting compiler.py emitter to import runtime symbols from the vendored namespace..."
# jinja2's compiler.py emits a literal `from jinja2.runtime import ...`
# line into every compiled template. Without this rewrite, compiled
# templates would either pull in the user's installed jinja2 (giving
# undefined behaviour if it's a different version) or fail entirely
# (if no system jinja2 is installed). We need them to resolve to the
# vendored runtime instead.
sed -i.bak \
    -e 's|from jinja2\.runtime import |from cyllama._vendor.jinja2.runtime import |' \
    "${VENDOR_DIR}/jinja2/compiler.py"
rm -f "${VENDOR_DIR}/jinja2/compiler.py.bak"

echo "Verifying no stray top-level jinja2 imports remain..."
if grep -rE '^from markupsafe import|^import markupsafe$' "${VENDOR_DIR}/jinja2/"; then
    echo "ERROR: found unrewritten markupsafe imports above" >&2
    exit 1
fi
if grep -rnE 'from jinja2\.runtime import|"jinja2\.runtime"' "${VENDOR_DIR}/jinja2/" | grep -v 'cyllama\._vendor\.jinja2\.runtime'; then
    echo "ERROR: found unrewritten jinja2.runtime references above" >&2
    exit 1
fi

echo "Copying LICENSE files..."
# Wheels from pip download include LICENSE files in dist-info under the
# `licenses/` subdirectory in modern wheels (PEP 639).
JINJA2_LICENSE=$(find "${WORK_DIR}/extracted" -path "*jinja2*dist-info*LICENSE*" -type f | head -1)
MARKUPSAFE_LICENSE=$(find "${WORK_DIR}/extracted" -path "*markupsafe*dist-info*LICENSE*" -type f | head -1)
[[ -n "${JINJA2_LICENSE}" ]] && cp "${JINJA2_LICENSE}" "${VENDOR_DIR}/LICENSE.jinja2"
[[ -n "${MARKUPSAFE_LICENSE}" ]] && cp "${MARKUPSAFE_LICENSE}" "${VENDOR_DIR}/LICENSE.markupsafe"

echo
echo "Re-vendoring complete."
echo "  jinja2:     ${JINJA2_VERSION}"
echo "  markupsafe: ${MARKUPSAFE_VERSION}"
echo
echo "Next steps:"
echo "  1. Update the version table in src/cyllama/_vendor/README.md"
echo "  2. Run the test suite: uv run pytest tests/test_jinja_chat.py"
echo "  3. Inspect the diff and commit"
