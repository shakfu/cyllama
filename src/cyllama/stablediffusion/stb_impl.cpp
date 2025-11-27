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

#include "stb_image.h"
#include "stb_image_write.h"
