#include "SceneConstants.hpp"

namespace tracer::renderer
{

namespace
{
using namespace containers;
}

SceneConstants::SceneConstants(const Vec3 vecX, const Vec3 vecZ, const Vec3 center, const Vec3 direction,
    const uint32_t samples, const uint32_t maxDepth, const uint32_t objectCount, const uint32_t width,
    const uint32_t height)
    : vecX_(vecX)
    , vecZ_(vecZ)
    , center_(center)
    , direction_(direction)
    , samples_(samples)
    , maxDepth_(maxDepth)
    , objectCount_(objectCount)
    , width_(width)
    , height_(height)
{}

} // namespace tracer::renderer
