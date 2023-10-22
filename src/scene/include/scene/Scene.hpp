#pragma once

#include <vector>

#include "objects/Sphere.hpp"
#include "utils/Ray.hpp"
#include "utils/Vec.hpp"

namespace tracer::scene
{

class Scene
{
public:
    Scene();

    void initScene();
    utils::Ray getCamera();
    int getObjectCount();
    objects::Sphere getObjectAt(int id);

private:
    int height_;
    int width_;
    std::vector<objects::Sphere> spheres_;
    utils::Ray camera_;
};

}  // namespace tracer::scene
