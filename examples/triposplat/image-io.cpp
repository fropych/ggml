#include "image-io.h"

#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"

#include <webp/decode.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace triposplat {
namespace {

std::vector<uint8_t> read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("cannot open input image: " + path);
    const auto size = input.tellg();
    if (size <= 0) throw std::runtime_error("input image is empty: " + path);
    input.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    input.read(reinterpret_cast<char *>(bytes.data()), size);
    if (!input) throw std::runtime_error("cannot read input image: " + path);
    return bytes;
}

bool is_webp(const std::vector<uint8_t> & bytes) {
    return bytes.size() >= 12 &&
        std::equal(bytes.begin(), bytes.begin() + 4, "RIFF") &&
        std::equal(bytes.begin() + 8, bytes.begin() + 12, "WEBP");
}

} // namespace

bool rgba_image::has_transparency() const {
    for (size_t i = 3; i < pixels.size(); i += 4) {
        if (pixels[i] != 255) return true;
    }
    return false;
}

rgba_image load_rgba_image(const std::string & path) {
    const std::vector<uint8_t> bytes = read_file(path);
    rgba_image image;
    uint8_t * decoded = nullptr;
    if (is_webp(bytes)) {
        decoded = WebPDecodeRGBA(bytes.data(), bytes.size(), &image.width, &image.height);
        if (!decoded) throw std::runtime_error("cannot decode WebP image: " + path);
        image.pixels.assign(decoded, decoded + size_t(image.width) * image.height * 4);
        WebPFree(decoded);
    } else {
        int channels = 0;
        decoded = stbi_load_from_memory(bytes.data(), int(bytes.size()),
                                        &image.width, &image.height, &channels, 4);
        if (!decoded) {
            throw std::runtime_error("cannot decode image " + path + ": " +
                                     (stbi_failure_reason() ? stbi_failure_reason() : "unknown error"));
        }
        image.pixels.assign(decoded, decoded + size_t(image.width) * image.height * 4);
        stbi_image_free(decoded);
    }
    if (image.width <= 0 || image.height <= 0) {
        throw std::runtime_error("decoded image has invalid dimensions");
    }
    return image;
}

rgba_image resize_image_bilinear(const rgba_image & source, int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("invalid resize dimensions");
    if (source.width == width && source.height == height) return source;
    rgba_image result {width, height, std::vector<uint8_t>(size_t(width) * height * 4)};
    const float sx = width > 1 ? float(source.width - 1) / float(width - 1) : 0.0f;
    const float sy = height > 1 ? float(source.height - 1) / float(height - 1) : 0.0f;
    for (int y = 0; y < height; ++y) {
        const float fy = y * sy;
        const int y0 = int(std::floor(fy)), y1 = std::min(y0 + 1, source.height - 1);
        const float wy = fy - y0;
        for (int x = 0; x < width; ++x) {
            const float fx = x * sx;
            const int x0 = int(std::floor(fx)), x1 = std::min(x0 + 1, source.width - 1);
            const float wx = fx - x0;
            for (int c = 0; c < 4; ++c) {
                const float top =
                    source.pixels[(size_t(y0) * source.width + x0) * 4 + c] * (1 - wx) +
                    source.pixels[(size_t(y0) * source.width + x1) * 4 + c] * wx;
                const float bottom =
                    source.pixels[(size_t(y1) * source.width + x0) * 4 + c] * (1 - wx) +
                    source.pixels[(size_t(y1) * source.width + x1) * 4 + c] * wx;
                result.pixels[(size_t(y) * width + x) * 4 + c] =
                    uint8_t(std::clamp(std::lround(top * (1 - wy) + bottom * wy), 0l, 255l));
            }
        }
    }
    return result;
}

rgba_image resize_short_side(const rgba_image & image, int size) {
    const float scale = float(size) / std::min(image.width, image.height);
    return resize_image_bilinear(
        image, std::max(1, int(std::lround(image.width * scale))),
        std::max(1, int(std::lround(image.height * scale))));
}

rgba_image prepare_foreground(rgba_image image, int canvas, int erode_radius) {
    if (erode_radius > 0) {
        std::vector<uint8_t> alpha(size_t(image.width) * image.height);
        for (int y = 0; y < image.height; ++y) for (int x = 0; x < image.width; ++x) {
            uint8_t value = 255;
            for (int dy = -erode_radius; dy <= erode_radius; ++dy) {
                const int yy = std::clamp(y + dy, 0, image.height - 1);
                for (int dx = -erode_radius; dx <= erode_radius; ++dx) {
                    const int xx = std::clamp(x + dx, 0, image.width - 1);
                    value = std::min(value, image.pixels[(size_t(yy) * image.width + xx) * 4 + 3]);
                }
            }
            alpha[size_t(y) * image.width + x] = value;
        }
        for (size_t i = 0; i < alpha.size(); ++i) image.pixels[i * 4 + 3] = alpha[i];
    }
    int min_x = image.width, min_y = image.height, max_x = -1, max_y = -1;
    for (int y = 0; y < image.height; ++y) for (int x = 0; x < image.width; ++x) {
        if (image.pixels[(size_t(y) * image.width + x) * 4 + 3] != 0) {
            min_x = std::min(min_x, x); max_x = std::max(max_x, x);
            min_y = std::min(min_y, y); max_y = std::max(max_y, y);
        }
    }
    if (max_x < 0) throw std::runtime_error("background removal produced an empty foreground");
    const float cx = (min_x + max_x) * 0.5f, cy = (min_y + max_y) * 0.5f;
    const float half = std::max(max_x - min_x, max_y - min_y) * 0.5f * 1.2f;
    const int left = int(cx - half), top = int(cy - half);
    const int crop_size = std::max(1, int(half * 2.0f));
    rgba_image crop {crop_size, crop_size, std::vector<uint8_t>(size_t(crop_size) * crop_size * 4)};
    for (int y = 0; y < crop_size; ++y) for (int x = 0; x < crop_size; ++x) {
        const int sx = left + x, sy = top + y;
        if (sx >= 0 && sx < image.width && sy >= 0 && sy < image.height) {
            std::copy_n(&image.pixels[(size_t(sy) * image.width + sx) * 4], 4,
                        &crop.pixels[(size_t(y) * crop_size + x) * 4]);
        }
    }
    rgba_image result = resize_image_bilinear(crop, canvas, canvas);
    for (size_t i = 0; i < size_t(canvas) * canvas; ++i) {
        const float alpha = result.pixels[i * 4 + 3] / 255.0f;
        for (int c = 0; c < 3; ++c) result.pixels[i * 4 + c] =
            uint8_t(std::lround(result.pixels[i * 4 + c] * alpha));
        result.pixels[i * 4 + 3] = 255;
    }
    return result;
}

} // namespace triposplat
