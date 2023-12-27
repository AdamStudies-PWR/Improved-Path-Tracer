#pragma once

#include <random>
#include <vector>

#include "containers/Vec3.hpp"
#include "scene/SceneData.hpp"

namespace tracer::renderer
{

class Renderer2
{
public:
    Renderer2(scene::SceneData& sceneData, const uint32_t samples);

    std::vector<containers::Vec3> render();

private:
    containers::Vec3 sendRay();

    const uint32_t samples_;
    scene::SceneData& sceneData_;
    std::mt19937 generator_;
};

}  // namespace tracer::renderer
