#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"
#include "scene/objects/AObject.hpp"
#include "scene/objects/Camera.hpp"

namespace tracer::scene
{

class SceneData
{
public:
    SceneData(const std::string& jsonPath);

    bool initScene();
    objects::Camera getCamera() const;
    std::shared_ptr<objects::AObject> getObjectAt(int id) const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    std::pair<int, double> getHitObjectAndDistance(const containers::Ray& ray) const;

private:
    bool loadBasicSceneData(const nlohmann::json& jsonData);
    bool loadCamera(const nlohmann::json& jsonData);
    bool loadObjects(const nlohmann::json& jsonData);
    bool addSpehere(const nlohmann::json& sphereData);
    bool addPlane(const nlohmann::json& planeData);

    const std::string jsonPath_;

    uint32_t height_;
    uint32_t width_;
    std::vector<std::shared_ptr<objects::AObject>> objects_;
    objects::Camera camera_;
};

}  // namespace tracer::scene
