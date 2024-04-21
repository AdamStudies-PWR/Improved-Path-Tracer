#pragma once

#include <cstdint>


namespace tracer::renderer
{

struct ImageData
{
    ImageData(const uint32_t width, const uint32_t height, const uint32_t samples, const uint8_t maxDepth,
        const uint32_t propCount, const uint32_t lightCount);

    const uint32_t width_;
    const uint32_t height_;
    const uint32_t samples_;
    const uint8_t maxDepth_;
    const uint32_t propCount_;
    const uint32_t lightCount_;
};

} // namespace tracer::renderer
