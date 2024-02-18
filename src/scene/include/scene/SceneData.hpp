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

namespace tracer::scene
{

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif

struct HitData
{
    __device__ HitData(int index, double distance)
        : index_(index)
        , distance_(distance)
    {}

    int index_;
    double distance_;
};

struct ObjectData
{
    ObjectData(const char objectType[], double radius, containers::Vec3 position, containers::Vec3 emission,
        containers::Vec3 color, objects::EReflectionType reflectionType);

    ObjectData(const char objectType[], containers::Vec3 north, containers::Vec3 east, containers::Vec3 position,
        containers::Vec3 emission, containers::Vec3 color, objects::EReflectionType reflectionType);

    const char* objectType_;
    double radius_;
    containers::Vec3 north_;
    containers::Vec3 east_;
    containers::Vec3 position_;
    containers::Vec3 emission_;
    containers::Vec3 color_;
    objects::EReflectionType reflectionType_;
};

class SceneData
{
public:
    SceneData(const std::string& jsonPath);

    bool initScene();

    // __device__ objects::AObject* getObjectAt(int id) { return objects_[id]; };

    __device__ HitData getHitObjectAndDistance(const containers::Ray& ray) const
    {
        int index = -1;
        double distance = 1e20;

        for (uint32_t i=0; i<objectCount_; i++)
        {
            // printf("Worke\n");
            auto temp = 0.0;
            /*if (objects_[i])
            {
                printf("Object exists :(\n");
                auto temp = objects_[i]->intersect(ray);
            }
            else
            {
                printf("Does not exist\n");
            }*/
            if (temp && temp < distance)
            {
                distance = temp;
                index = i;
            }
        }

        return HitData(index, distance);
    }

    __host__ __device__ objects::Camera getCamera() const { return camera_; }
    __host__ __device__ uint32_t getWidth() const { return width_; }
    __host__ __device__ uint32_t getHeight() const { return height_; }
    __device__ ObjectData** getObjectsData() const
    {
        for (uint32_t i=0; i<objectCount_; i++)
        {
            printf("Check2: %s\n", objectsData_[i]->objectType_);
        }
        return objectsData_; }
    __host__ __device__ uint32_t getObjectCount() const { return objectCount_; }

private:
    bool loadBasicSceneData(const nlohmann::json& jsonData);
    bool loadCamera(const nlohmann::json& jsonData);
    bool loadObjects(const nlohmann::json& jsonData);
    bool addSpehere(const nlohmann::json& sphereData, std::vector<ObjectData*>& container);
    bool addPlane(const nlohmann::json& planeData, std::vector<ObjectData*>& container);

    const std::string jsonPath_;
    objects::Camera camera_;
    std::map<std::string, bool (SceneData::*)(const nlohmann::json&, std::vector<ObjectData*>&)> typeToHandler_;
    ObjectData** objectsData_;
    uint32_t objectCount_;
    uint32_t height_;
    uint32_t width_;
};

}  // namespace tracer::scene
