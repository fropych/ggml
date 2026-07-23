#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace triposplat {

struct rgba_image {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;

    bool has_transparency() const;
};

rgba_image load_rgba_image(const std::string & path);
rgba_image resize_image_bilinear(const rgba_image & image, int width, int height);
rgba_image resize_short_side(const rgba_image & image, int size);
rgba_image prepare_foreground(rgba_image image, int canvas, int erode_radius);

} // namespace triposplat
