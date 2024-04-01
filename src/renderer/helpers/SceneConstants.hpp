#pragma once

#include <cstdint>

#include "containers/Vec3.hpp"


namespace tracer::renderer
{

struct SceneConstants
{
    SceneConstants(const containers::Vec3 vecX, const containers::Vec3 vecZ, const containers::Vec3 center,
        const containers::Vec3 direction, const uint32_t samples, const uint32_t maxDepth, const uint32_t objectCount,
        const uint32_t width, const uint32_t height);

    const containers::Vec3 vecX_;
    const containers::Vec3 vecZ_;
    const containers::Vec3 center_;
    const containers::Vec3 direction_;
    const uint32_t samples_;
    const uint32_t maxDepth_;
    const uint32_t objectCount_;
    const uint32_t width_;
    const uint32_t height_;
};

} // namespace tracer::renderer
