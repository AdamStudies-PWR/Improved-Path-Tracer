#pragma once

#include <cstdint>
#include <vector>

#include "containers/Vec3.hpp"

namespace tracer::utils
{

void saveImage(containers::Vec3* image, int height, int width);
void saveImage(const std::vector<containers::Vec3>& image, const uint32_t height, const uint32_t width);

}  // namespace tracer::utils
