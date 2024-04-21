#include "ImageData.hpp"

namespace tracer::renderer
{

ImageData::ImageData(const uint32_t width, const uint32_t height, const uint32_t samples, const uint8_t maxDepth,
    const uint32_t propCount, const uint32_t lightCount)
    : width_(width)
    , height_(height)
    , samples_(samples)
    , maxDepth_(maxDepth)
    , propCount_(propCount)
    , lightCount_(lightCount)
{}

} // namespace tracer::renderer
