#pragma once

#include "containers/Vec3.hpp"

namespace tracer::renderer
{

struct PixelData
{
    PixelData(const double stepX, const double stepZ, const containers::Vec3 gaze);

    const double stepX_;
    const double stepZ_;
    const containers::Vec3 gaze_;
};

}  // namespace tracer::renderer
