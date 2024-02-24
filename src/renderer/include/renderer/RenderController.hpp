#pragma once

#include <vector>

#include "containers/Vec3.hpp"
#include "scene/SceneData.hpp"

namespace tracer::renderer
{

class RenderContoller
{
public:
    RenderContoller(scene::SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth);

    std::vector<containers::Vec3> start();

private:
    std::vector<containers::Vec3> convertToVector(containers::Vec3* imagePtr);

    scene::SceneData& sceneData_;
    const uint8_t maxDepth_;
    const uint32_t samples_;
};

}  // namespace tracer::renderer
