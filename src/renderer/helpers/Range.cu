#include <cstdint>

namespace tracer::renderer
{

struct Range
{
    __device__ Range(uint32_t startX, uint32_t startZ, uint32_t endX, uint32_t endZ)
        : startX_(startX)
        , startZ_(startZ)
        , endX_(endX)
        , endZ_(endZ)
    {}

    const uint32_t startX_;
    const uint32_t startZ_;
    const uint32_t endX_;
    const uint32_t endZ_;
};

}  // namespace tracer::renderer
