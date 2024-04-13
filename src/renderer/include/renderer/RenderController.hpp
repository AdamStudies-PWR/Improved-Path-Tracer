#pragma once

#include <vector>

#include "containers/Vec3.hpp"
#include "objects/AObject.hpp"
#include "scene/SceneData.hpp"


namespace tracer::renderer
{

class PixelData;
class SceneConstants;

class RenderController
{
public:
    RenderController(scene::SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth);

    std::vector<containers::Vec3> start();
private:
    void renderGPU(const uint32_t z, scene::objects::AObject** devObjects, SceneConstants* devConstants,
        const containers::Vec3& vecZ);
    containers::Vec3 startKernel(scene::objects::AObject** devObjects, containers::Vec3* devSamples,
        PixelData* devPixelData, SceneConstants* devConstants, const containers::Vec3& vecZ, const uint32_t pixelX,
        const uint32_t pixelZ);

    scene::SceneData& sceneData_;
    std::vector<containers::Vec3> image_;
    const uint8_t maxDepth_;
    const uint32_t samples_;
};

}  // namespace tracer::renderer
