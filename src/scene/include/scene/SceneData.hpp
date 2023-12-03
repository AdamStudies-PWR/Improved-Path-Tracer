#pragma once

#include <string>
#include <vector>
#include <memory>

#include "scene/objects/AObject.hpp"
#include "containers/Ray.hpp"
#include "containers/Vec.hpp"

namespace tracer::scene
{

class SceneData
{
public:
    SceneData(const std::string& jsonPath);

    bool initScene();
    containers::Ray getCamera();
    int getObjectCount();
    std::shared_ptr<objects::AObject> getObjectAt(int id);

    // TEMP
    std::vector<std::shared_ptr<objects::AObject>> objects_;

private:
    const std::string jsonPath_;
    int height_;
    int width_;
    // std::vector<objects::Sphere> objects_;
    containers::Ray camera_;
};

}  // namespace tracer::scene
