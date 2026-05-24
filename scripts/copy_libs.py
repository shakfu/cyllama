#!/usr/bin/env python3
"""
Script to copy:
1. All .a and .lib files from thirdparty/llama.cpp/build to src/llama.cpp/lib
2. All .h, .hpp files from thirdparty/llama.cpp to src/llama.cpp/include
3. All .cpp, .cc files from thirdparty/llama.cpp to src/llama.cpp/src
"""

import os
import shutil
import glob
import logging
import platform

ROOT = os.path.join(os.path.dirname(__file__), "..")


def copy_library_files():
    # Define source and destination directories
    src_dir = os.path.join(ROOT, "thirdparty", "llama.cpp", "build")
    dst_dir = os.path.join(ROOT, "src", "llama.cpp", "lib")

    # Create destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        logging.info(f"Creating destination directory: {dst_dir}")
        os.makedirs(dst_dir, exist_ok=True)

    # Recursively find all static library files
    lib_files = []
    for ext in (".a", ".lib"):
        pattern = os.path.join(src_dir, "**", f"*{ext}")
        lib_files.extend(glob.glob(pattern, recursive=True))

    if not lib_files:
        logging.warning(f"No .a or .lib files found in {src_dir}")
        return

    staged_count = 0
    skipped_count = 0

    for lib_file in lib_files:
        filename = os.path.basename(lib_file)
        dst_file = os.path.join(dst_dir, filename)

        # Skip if link or file already exists
        if os.path.exists(dst_file):
            logging.info(f"Skipping {filename} - already exists at {dst_file}")
            skipped_count += 1
            continue

        if platform.system() == "Windows":
            logging.info(f"Copying {lib_file} -> {dst_file}")
            shutil.copy2(lib_file, dst_file)
        else:
            try:
                logging.info(f"Linking {lib_file} -> {dst_file}")
                os.symlink(lib_file, dst_file)
            except OSError:
                logging.info(f"Symlink failed; copying {lib_file} -> {dst_file}")
                shutil.copy2(lib_file, dst_file)
        staged_count += 1

    logging.info(
        f"Successfully staged {staged_count} libraries to {dst_dir} "
        f"({skipped_count} skipped)"
    )


def copy_source_files(target, source_paths):
    # Define source base directory and destination directory
    src_base = os.path.join(ROOT, "thirdparty", "llama.cpp")
    dst_dir = os.path.join(ROOT, "src", "llama.cpp", target)

    # Create destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        logging.info(f"Creating destination directory: {dst_dir}")
        os.makedirs(dst_dir, exist_ok=True)

    # Copy each file to destination
    copied_count = 0
    skipped_count = 0
    for rel_path in source_paths:
        src_file = os.path.join(src_base, rel_path)

        # Skip if source file doesn't exist
        if not os.path.exists(src_file):
            logging.warning(f"Source file not found: {src_file}")
            continue

        # Create subdirectories in destination if needed
        filename = os.path.basename(rel_path)
        dst_file = os.path.join(dst_dir, filename)

        # Skip if destination file already exists
        if os.path.exists(dst_file):
            logging.info(f"Skipping {filename} - already exists at {dst_file}")
            skipped_count += 1
            continue

        logging.info(f"Copying {src_file} to {dst_file}")
        shutil.copy2(src_file, dst_file)
        copied_count += 1

    logging.info(
        f"Successfully copied {copied_count} source files to {dst_dir} ({skipped_count} skipped)"
    )


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    # Copy library files
    copy_library_files()

    # Copy header files
    copy_source_files(
        "include",
        [
            "common/common.h",
            "ggml/include/ggml.h",
            "ggml/include/ggml-backend.h",
            "include/llama.h",
        ],
    )
    copy_source_files(
        "src",
        [
            "tools/server/server.cpp",
        ],
    )


if __name__ == "__main__":
    main()
