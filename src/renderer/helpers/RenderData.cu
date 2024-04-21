#include <cstdint>

#include "objects/AObject.hpp"


namespace tracer::renderer
{

struct RenderData
{
    __device__ RenderData(scene::objects::AObject** props, scene::objects::AObject** lights, const uint32_t propCount,
        const uint32_t lightCount, const uint32_t maxDepth, curandState& state)
        : props_(props)
        , lights_(lights)
        , propCount_(propCount)
        , lightCount_(lightCount)
        , maxDepth_(maxDepth)
        , state_(state)
    {}

    scene::objects::AObject** props_;
    scene::objects::AObject** lights_;
    const uint32_t propCount_;
    const uint32_t lightCount_;
    const uint32_t maxDepth_;
    curandState& state_;
};

}  // namespace tracer::renderer
