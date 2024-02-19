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

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif

struct HitData
{
    HitData(int index, double distance)
        : index_(index)
        , distance_(distance)
    {}

    int index_;
    double distance_;
};

class SceneData
{
public:
    SceneData(const std::string& jsonPath);

    bool initScene();

    // __device__ objects::AObject* getObjectAt(int id) { return objects_[id]; };

    HitData getHitObjectAndDistance(const containers::Ray& /*ray*/) const
    {
        int index = -1;
        double distance = 1e20;

        for (uint32_t i=0; i<objectsData_.size(); i++)
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
    std::vector<objects::ObjectData*> getObjectsData() const { return objectsData_; }

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
