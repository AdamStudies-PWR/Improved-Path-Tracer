#pragma once

#include <vector>

#include "containers/Vec3.hpp"
#include "objects/AObject.hpp"
#include "scene/SceneData.hpp"


namespace tracer::renderer
{

class SceneConstants;

class RenderController
{
public:
    RenderController(scene::SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth);

    std::vector<containers::Vec3> start();
private:
    void renderGPU(const uint32_t z, scene::objects::AObject** devObjects, SceneConstants* devConstants);
    void startKernel(containers::Vec3* devRow, scene::objects::AObject** devObjects, SceneConstants* devConstants,
        const uint32_t z);

    scene::SceneData& sceneData_;
    const uint8_t maxDepth_;
    const uint32_t samples_;
    std::vector<containers::Vec3> image_;
};

}  // namespace tracer::renderer
