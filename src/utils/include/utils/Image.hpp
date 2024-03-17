#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "containers/Vec3.hpp"

namespace tracer::utils
{

void saveImage(const std::vector<containers::Vec3>& image, const uint32_t height, const uint32_t width,
    const std::string filename);

}  // namespace tracer::utils
