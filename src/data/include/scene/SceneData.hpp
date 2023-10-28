#pragma once

#include <vector>

#include "objects/Sphere.hpp"
#include "containers/Ray.hpp"
#include "containers/Vec.hpp"

namespace tracer::data
{

class SceneData
{
public:
    SceneData();

    void initScene();
    containers::Ray getCamera();
    int getObjectCount();
    objects::Sphere getObjectAt(int id);

private:
    int height_;
    int width_;
    std::vector<objects::Sphere> spheres_;
    containers::Ray camera_;
};

}  // namespace tracer::data
