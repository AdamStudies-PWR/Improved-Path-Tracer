#include "scene/SceneData.hpp"

#include <fstream>
#include <iostream>

#include "objects/Plane.hpp"
#include "objects/Sphere.hpp"

namespace tracer::scene
{

namespace
{
using json = nlohmann::json;
using namespace containers;
using namespace objects;

const double INF = 1e20;

json loadJsonFile(const std::string filename)
{
    std::ifstream file(filename);
    if (not file)
    {
        return nullptr;
    }

    json data = json::parse(file);
    file.close();

    return data;
}

bool validateVec3tor(const json& vecData)
{
    return vecData.contains("xx") && vecData.contains("yy") && vecData.contains("yy");
}

bool validateObject(const json& objectData)
{
    if (not objectData.contains("color") || not objectData.contains("emission") || not objectData.contains("position")
        || not objectData.contains("reflection") || not objectData.contains("type"))
    {
        return false;
    }

    if (not validateVec3tor(objectData["color"]) || not validateVec3tor(objectData["emission"])
        || not validateVec3tor(objectData["position"]))
    {
        return false;
    }

    return true;
}
}  // namespace

SceneData::SceneData(const std::string& jsonPath)
    : jsonPath_(jsonPath)
{
    typeToHandler_["sphere"] = &SceneData::addSpehere;
    typeToHandler_["plane"] = &SceneData::addPlane;
}

bool SceneData::initScene()
{
    std::cout << "Loading Scene Data..." << std::endl;

    const auto jsonData = loadJsonFile(jsonPath_);
    if (jsonData == nullptr)
    {
        std::cout << "Could not load provided json file!" << std::endl;
        return false;
    }

    if (not loadBasicSceneData(jsonData))
    {
        return false;
    }

    if (not loadCamera(jsonData))
    {
        return false;
    }

    if (not loadObjects(jsonData))
    {
        return false;
    }

    if (objects_.empty())
    {
        std::cout << "Object list empty! Cannot build scene" << std::endl;
        return false;
    }

    std::cout << "Data loaded successfully" << std::endl;

    return true;
}

Camera SceneData::getCamera() const { return camera_; }
std::shared_ptr<AObject> SceneData::getObjectAt(int id) const { return objects_.at(id); }
uint32_t SceneData::getWidth() const { return width_; }
uint32_t SceneData::getHeight() const { return height_; }

std::pair<int, double> SceneData::getHitObjectAndDistance(const Ray& ray) const
{
    int index = -1;
    double distance = INF;

    for (const auto& object : objects_)
    {
        auto temp = object->intersect(ray);
        if (temp && temp < distance)
        {
            distance = temp;
            index = &object - &objects_[0];
        }
    }

    return {index, distance};
}

bool SceneData::loadBasicSceneData(const json& jsonData)
{
    if (not jsonData.contains("height") || not jsonData.contains("width"))
    {
        std::cout << "Missing height or witdh data!" << std::endl;
        return false;
    }

    height_ = jsonData["height"];
    width_ = jsonData["width"];

    return true;
}

bool SceneData::loadCamera(const nlohmann::json& jsonData)
{
    if (not jsonData.contains("camera"))
    {
        std::cout << "No camera data!" << std::endl;
        return false;
    }

    const auto cameraData = jsonData["camera"];
    if (not cameraData.contains("direction") or not cameraData.contains("position")
        or not cameraData.contains("orientation"))
    {
        std::cout << "Camera data could not be read!" << std::endl;
        return false;
    }

    const auto directionData = cameraData["direction"];
    const auto positionData = cameraData["position"];
    const auto orientationData = cameraData["orientation"];

    if (not validateVec3tor(directionData) or not validateVec3tor(positionData) or not validateVec3tor(orientationData))
    {
        std::cout << "Camera data could not be parsed!" << std::endl;
        return false;
    }

    camera_ = Camera(Vec3(positionData["xx"], positionData["yy"], positionData["zz"]),
                  Vec3(directionData["xx"], directionData["yy"], directionData["zz"]).norm(),
                  Vec3(orientationData["xx"], orientationData["yy"], orientationData["zz"]).norm());

    return true;
}

bool SceneData::loadObjects(const nlohmann::json& jsonData)
{
    if (not jsonData.contains("objects"))
    {
        std::cout << "No objects data!" << std::endl;
        return false;
    }

    for (const auto& object : jsonData["objects"])
    {
        if (not validateObject(object))
        {
            std::cout << "Could not validate object data!" << std::endl;
            return false;
        }

        if (typeToHandler_[object["type"]])
        {
            if (not (this->*typeToHandler_[object["type"]])(object))
            {
                return false;
            }
        }
        else
        {
            std::cout << "Unknown object type" << std::endl;
            return false;
        }
    }

    return true;
}

bool SceneData::addSpehere(const json& sphereData)
{
    if (not sphereData.contains("radius"))
    {
        std::cout << "Broken sphere object! " << std::endl;
        return false;
    }

    const auto position = sphereData["position"];
    const auto color = sphereData["color"];
    const auto emission = sphereData["emission"];

    objects_.push_back(std::make_shared<Sphere>(sphereData["radius"],
                                                Vec3(position["xx"], position["yy"], position["zz"]),
                                                Vec3(emission["xx"], emission["yy"], emission["zz"]),
                                                Vec3(color["xx"], color["yy"], color["zz"]),
                                                EReflectionType(sphereData["reflection"])));

    return true;
}

bool SceneData::addPlane(const json& planeData)
{
    if (not planeData.contains("north") or not planeData.contains("east"))
    {
        std::cout << "Broken plane object! " << std::endl;
        return false;
    }

    const auto north = planeData["north"];
    const auto east = planeData["east"];
    const auto position = planeData["position"];
    const auto color = planeData["color"];
    const auto emission = planeData["emission"];

    objects_.push_back(std::make_shared<Plane>(Vec3(north["xx"], north["yy"], north["zz"]),
                                               Vec3(east["xx"], east["yy"], east["zz"]),
                                               Vec3(position["xx"], position["yy"], position["zz"]),
                                               Vec3(emission["xx"], emission["yy"], emission["zz"]),
                                               Vec3(color["xx"], color["yy"], color["zz"]),
                                               EReflectionType(planeData["reflection"])));

    return true;
}

}  // namespace tracer::scene
