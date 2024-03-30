#include <cstdint>

namespace tracer::renderer
{

struct LoopRange
{
    __device__ LoopRange(const uint32_t start, const uint32_t stop)
        : start_(start)
        , stop_(stop)
    {}

    const uint32_t start_;
    const uint32_t stop_;
};

}  // namespace tracer::renderer
