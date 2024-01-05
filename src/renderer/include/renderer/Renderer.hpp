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
    Renderer(scene::SceneData& sceneData, const uint32_t samples);

    std::vector<containers::Vec3> render();

private:
    containers::Vec3 samplePixel(const containers::Vec3 stepX, const containers::Vec3 stepY, const uint32_t pixelX,
        const uint32_t pixelY);
    containers::Vec3 sendRay(const containers::Ray& ray, uint16_t depth);

    const uint32_t samples_;
    scene::SceneData& sceneData_;
    std::mt19937 generator_;
};

}  // namespace tracer::renderer
