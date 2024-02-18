#pragma once

#include <random>
#include <vector>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"
#include "scene/SceneData.hpp"

namespace tracer::renderer
{

#ifndef __device__
#define __device__
#endif

struct Coordinates
{
    __device__ Coordinates(uint32_t xx, uint32_t zz, uint32_t loopX, uint32_t loopZ);

    uint32_t xx_;
    uint32_t zz_;
    uint32_t loopX_;
    uint32_t loopZ_;
};

class Renderer
{
public:
    Renderer(scene::SceneData& sceneData, const uint32_t samples);

    std::vector<containers::Vec3> render();

private:
    containers::Vec3 samplePixel2(const containers::Vec3& vecX, const containers::Vec3& vecZ, const uint32_t pixelX,
        const uint32_t pixelZ);
    containers::Vec3 sendRay(const containers::Ray& ray, uint8_t depth);

    const uint32_t samples_;
    scene::SceneData& sceneData_;
    std::mt19937 generator_;
};

}  // namespace tracer::renderer
