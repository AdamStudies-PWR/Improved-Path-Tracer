#pragma once

#include <vector>

#include "containers/Vec3.hpp"
#include "objects/AObject.hpp"
#include "scene/objects/Camera.hpp"
#include "scene/SceneData.hpp"


namespace tracer::renderer
{

class PixelData;
class SceneConstants;

class RenderController
{
public:
    RenderController(scene::SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth);

    std::vector<containers::Vec3> start(const std::vector<scene::objects::ObjectData>& objectDataVec);
private:
    void renderGPU(const uint32_t z, scene::objects::AObject** devObjects, SceneConstants* devConstants,
        const containers::Vec3& vecZ);
    containers::Vec3 startKernel(scene::objects::AObject** devObjects, containers::Vec3* devSamples,
        containers::Vec3* samplesPtr, PixelData* devPixelData, SceneConstants* devConstants,
        const containers::Vec3& vecZ, const uint32_t pixelX, const uint32_t pixelZ);

    scene::objects::Camera camera_;
    const uint32_t height_;
    const uint32_t samples_;
    const uint32_t width_;
    const uint8_t maxDepth_;
    const float correctionX_;
    const float correctionZ_;
    std::vector<containers::Vec3> image_;
};

}  // namespace tracer::renderer
