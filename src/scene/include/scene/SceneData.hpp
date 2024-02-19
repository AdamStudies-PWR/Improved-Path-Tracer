#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "containers/Ray.hpp"
#include "containers/Vec3.hpp"
#include "scene/objects/Camera.hpp"
#include "scene/objects/EReflectionType.hpp"
#include "scene/objects/ObjectData.hpp"

namespace tracer::scene
{

class SceneData
{
public:
    SceneData(const std::string& jsonPath);

    bool initScene();

    objects::Camera getCamera() const;
    uint32_t getWidth() const;
    uint32_t getHeight() const;
    std::vector<objects::ObjectData*> getObjectsData() const;

private:
    bool loadBasicSceneData(const nlohmann::json& jsonData);
    bool loadCamera(const nlohmann::json& jsonData);
    bool loadObjects(const nlohmann::json& jsonData);
    bool addSpehere(const nlohmann::json& sphereData);
    bool addPlane(const nlohmann::json& planeData);

    const std::string jsonPath_;
    objects::Camera camera_;
    std::map<std::string, bool (SceneData::*)(const nlohmann::json&)> typeToHandler_;
    std::vector<objects::ObjectData*> objectsData_;
    uint32_t height_;
    uint32_t width_;
};

}  // namespace tracer::scene
