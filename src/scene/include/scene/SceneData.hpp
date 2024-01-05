#pragma once

#include <string>
#include <vector>
#include <memory>

#include <nlohmann/json.hpp>

#include "scene/objects/AObject.hpp"
#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"

namespace tracer::scene
{

class SceneData
{
public:
    SceneData(const std::string& jsonPath);

    bool initScene();
    containers::Ray getCamera() const;
    int hasHitObject(const containers::Ray& ray) const;
    std::shared_ptr<objects::AObject> getObjectAt(int id) const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;

private:
    bool loadBasicSceneData(const nlohmann::json& jsonData);
    bool loadCamera(const nlohmann::json& jsonData);
    bool loadObjects(const nlohmann::json& jsonData);
    bool addSpehere(const nlohmann::json& sphereData);

    const std::string jsonPath_;

    uint32_t height_;
    uint32_t width_;
    std::vector<std::shared_ptr<objects::AObject>> objects_;
    containers::Ray camera_;
};

}  // namespace tracer::scene
