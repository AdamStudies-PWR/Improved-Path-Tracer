#include <stdio.h>

#include "containers/Vec3.hpp"
#include "scene/objects/AObject.hpp"
#include "scene/objects/ObjectData.hpp"
#include "scene/objects/Plane.cu"
#include "scene/objects/Sphere.cu"


namespace tracer::renderer
{

namespace
{
using namespace containers;
using namespace scene;
using namespace scene::objects;
}  // namespace

class Renderer
{
public:
    __device__ Renderer(AObject** objects)
        : objects_(objects)
        , objectsCount_(0)
    {}

    __device__ void setUp(ObjectData** objectsData, const uint32_t objectCount)
    {
        objectsCount_ = objectCount;
        for (uint32_t i=0; i<objectsCount_; i++)
        {
            if (objectsData[i]->objectType_ == SphereData)
            {
                auto* sphere = new Sphere(objectsData[i]->radius_, objectsData[i]->position_, objectsData[i]->emission_,
                    objectsData[i]->color_, objectsData[i]->reflectionType_);
                objects_[i] = sphere;
            }
            else if (objectsData[i]->objectType_ == PlaneData)
            {
                // printf("Plane\n");
                // auto* plane = new Plane(objectsData[i]->north_, objectsData[i]->east_, objectsData[i]->position_,
                //     objectsData[i]->emission_, objectsData[i]->color_, objectsData[i]->reflectionType_);
                // objects_[i] = plane;
            }
        }
    }

    __device__ void start()
    {}

private:
    AObject** objects_;
    uint32_t objectsCount_;
};

__global__ void cudaMain(Vec3* image, AObject** objects, ObjectData** objectsData, const uint32_t objectsCount)
{
    Renderer render = Renderer(objects);
    render.setUp(objectsData, objectsCount);
    printf("Finished setup\n");
    render.start();
}

}  // namespace tracer::render
