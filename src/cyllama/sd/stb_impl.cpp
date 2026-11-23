/*
 * stb_impl.cpp - Implementation file for stb_image libraries
 *
 * This file provides the implementation for stb_image and stb_image_write.
 * It must be compiled as part of the stable_diffusion extension.
 */

// stb_image_write uses std::min in C++ mode, so we need <algorithm>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
// Paths are passed as UTF-8; on Windows stb otherwise opens them as ANSI.
#define STBI_WINDOWS_UTF8
#define STBIW_WINDOWS_UTF8
// Refuse images over 32768 px per side before allocating (stb's default is 2^24).
#define STBI_MAX_DIMENSIONS (1 << 15)

#include "stb_image.h"
#include "stb_image_write.h"
