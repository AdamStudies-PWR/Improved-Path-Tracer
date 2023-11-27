#pragma once

#include <string>
#include <vector>

#include "scene/objects/Sphere.hpp"
#include "containers/Ray.hpp"
#include "containers/Vec.hpp"

namespace tracer::data
{

class SceneData
{
public:
    SceneData() = default;

    void initScene(const std::string& jsonPath);
    containers::Ray getCamera();
    int getObjectCount();
    objects::Sphere getObjectAt(int id);

    // TEMP
    std::vector<objects::Sphere> spheres_;

private:
    int height_;
    int width_;
    // std::vector<objects::Sphere> spheres_;
    containers::Ray camera_;
};

}  // namespace tracer::data
