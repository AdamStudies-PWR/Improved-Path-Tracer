#pragma once

#include <random>
#include <vector>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"
#include "scene/SceneData.hpp"

namespace tracer::renderer
{

class Renderer
{
public:
    Renderer(scene::SceneData& sceneData, const uint32_t samples, const uint8_t maxDepth);

    std::vector<containers::Vec3> render();

private:
    containers::Vec3 samplePixel(const containers::Vec3& vecX, const containers::Vec3& vecZ, const uint32_t pixelX,
        const uint32_t pixelZ);
    containers::Vec3 sendRay(const containers::Ray& ray, uint8_t depth);

    const uint8_t maxDepth_;
    const uint32_t samples_;
    scene::SceneData& sceneData_;
    std::mt19937 generator_;
};

}  // namespace tracer::renderer
