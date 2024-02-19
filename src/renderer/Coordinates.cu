#include <cstdint>

namespace tracer::renderer
{

struct Coordinates
{
    __device__ Coordinates(uint32_t xx, uint32_t zz, uint32_t loopX, uint32_t loopZ)
        : xx_(xx)
        , zz_(zz)
        , loopX_(loopX)
        , loopZ_(loopZ)
    {}

    uint32_t xx_;
    uint32_t zz_;
    uint32_t loopX_;
    uint32_t loopZ_;
};

}  // namespace tracer::renderer
