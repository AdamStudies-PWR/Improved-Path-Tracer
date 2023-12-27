#include "scene/SceneData.hpp"

#include <fstream>
#include <iostream>

#include "objects/Sphere.hpp"

namespace tracer::scene
{

namespace
{
using json = nlohmann::json;
using namespace containers;
using namespace objects;

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
{}

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

Ray SceneData::getCamera() const { return camera_; }
int SceneData::getObjectCount() const { return objects_.size(); }
std::shared_ptr<AObject> SceneData::getObjectAt(int id) const { return objects_.at(id); }
uint32_t SceneData::getWidth() const { return width_; }
uint32_t SceneData::getHeight() const { return height_; }

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
    if (not cameraData.contains("direction") || not cameraData.contains("position"))
    {
        std::cout << "Camera data contains no direction or position!" << std::endl;
        return false;
    }

    const auto directionData = cameraData["direction"];
    const auto positionData = cameraData["position"];

    if (not validateVec3tor(directionData) || not validateVec3tor(positionData))
    {
        std::cout << "Damaged position or direction vector!" << std::endl;
        return false;
    }

    camera_ = Ray(Vec3(positionData["xx"], positionData["yy"], positionData["zz"]),
                  Vec3(directionData["xx"], directionData["yy"], directionData["zz"]));

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

        if (object["type"] == "sphere")
        {
            if (not addSpehere(object))
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
        std::cout << " Broken sphere object! " << std::endl;
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

}  // namespace tracer::scene

// camera_ = Ray(Vec3(50, 52, 295.6), Vec3(0, -0.042612, -1).norm());
// objects_.push_back(std::make_shared<Sphere>(1e5, Vec3(1e5-9, 40.8, 81.6), Vec3(), Vec3(0.75, 0.25, 0.25), Diffuse));       // Left Wall
// objects_.push_back(std::make_shared<Sphere>(1e5, Vec3(-1e5+109, 40.8, 81.6), Vec3(), Vec3(0.25, 0.25, 0.75), Diffuse));    // Right Wall
// objects_.push_back(std::make_shared<Sphere>(1e5, Vec3(50, 40.8, 1e5), Vec3(), Vec3(0.75, 0.75, 0.75), Diffuse));           // Back Wall
// objects_.push_back(std::make_shared<Sphere>(1e5, Vec3(50, 40.8, -1e5+175), Vec3(), Vec3(0, 0.44, 0), Diffuse));            // Wall behind camera?
// objects_.push_back(std::make_shared<Sphere>(1e5, Vec3(50, 1e5, 81.6), Vec3(), Vec3(0.75, 0.75, 0.75), Diffuse));           // Floor
// objects_.push_back(std::make_shared<Sphere>(1e5, Vec3(50, -1e5+81.6, 81.6), Vec3(), Vec3(0.75, 0.75, 0.75), Diffuse));     // Ceiling
// objects_.push_back(std::make_shared<Sphere>(16.5, Vec3(27, 16.5, 47), Vec3(), Vec3(1, 1, 1) * 0.999, Specular));           // Left Orb (Mirror like)
// objects_.push_back(std::make_shared<Sphere>(16.5, Vec3(73, 16.5, 78), Vec3(), Vec3(1, 1, 1) * 0.999, Refractive));         // Right Orb (Glass ?)
// objects_.push_back(std::make_shared<Sphere>(600, Vec3(50, 681.6-.27, 81.6), Vec3(12, 12, 12), Vec3(), Diffuse));           // Light source
