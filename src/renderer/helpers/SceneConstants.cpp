#include "SceneConstants.hpp"

namespace tracer::renderer
{

namespace
{
using namespace containers;
}

SceneConstants::SceneConstants(const Vec3 vecX, const Vec3 vecZ, const Vec3 center, const Vec3 direction,
    const uint32_t samples, const uint32_t maxDepth, const uint32_t objectCount)
    : vecX_(vecX)
    , vecZ_(vecZ)
    , center_(center)
    , direction_(direction)
    , samples_(samples)
    , maxDepth_(maxDepth)
    , objectCount_(objectCount)
{}

} // namespace tracer::renderer
