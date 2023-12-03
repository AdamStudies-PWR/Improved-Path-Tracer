#include "scene/SceneData.hpp"

#include <iostream>

#include "objects/Sphere.hpp"

namespace tracer::scene
{

namespace
{
using namespace containers;
using namespace objects;
}

void SceneData::initScene(const std::string& jsonPath)
{
    std::cout << "Initilizing scene..." << std::endl;

    camera_ = Ray(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());

    objects_.push_back(std::make_shared<Sphere>(1e5, Vec(1e5-9, 40.8, 81.6), Vec(), Vec(0.75, 0.25, 0.25), Diffuse));       // Left Wall
    objects_.push_back(std::make_shared<Sphere>(1e5, Vec(-1e5+109, 40.8, 81.6), Vec(), Vec(0.25, 0.25, 0.75), Diffuse));    // Right Wall
    objects_.push_back(std::make_shared<Sphere>(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(0.75, 0.75, 0.75), Diffuse));           // Back Wall
    objects_.push_back(std::make_shared<Sphere>(1e5, Vec(50, 40.8, -1e5+175), Vec(), Vec(0, 0.44, 0), Diffuse));            // Wall behind camera?
    objects_.push_back(std::make_shared<Sphere>(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(0.75, 0.75, 0.75), Diffuse));           // Floor
    objects_.push_back(std::make_shared<Sphere>(1e5, Vec(50, -1e5+81.6, 81.6), Vec(), Vec(0.75, 0.75, 0.75), Diffuse));     // Ceiling
    objects_.push_back(std::make_shared<Sphere>(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * 0.999, Specular));           // Left Orb (Mirror like)
    objects_.push_back(std::make_shared<Sphere>(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * 0.999, Refractive));         // Right Orb (Glass ?)
    objects_.push_back(std::make_shared<Sphere>(600, Vec(50, 681.6-.27, 81.6), Vec(12, 12, 12), Vec(), Diffuse));           // Light source

    std::cout << "Done." << std::endl;
}

Ray SceneData::getCamera() { return camera_; }
int SceneData::getObjectCount() { return objects_.size(); }
std::shared_ptr<AObject> SceneData::getObjectAt(int id) { return objects_.at(id); }

}  // namespace tracer::scene
