#include "ImageData.hpp"

namespace tracer::renderer
{

ImageData::ImageData(const uint32_t width, const uint32_t height, const uint32_t samples, const uint8_t maxDepth,
    const uint32_t objectCount)
    : width_(width)
    , height_(height)
    , samples_(samples)
    , maxDepth_(maxDepth)
    , objectCount_(objectCount)
{}

} // namespace tracer::renderer
