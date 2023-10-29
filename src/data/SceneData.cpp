#include "scene/SceneData.hpp"

#include <iostream>

namespace tracer::data
{

SceneData::SceneData()
    : camera_(containers::Ray(containers::Vec(50, 52, 295.6), containers::Vec(0, -0.042612, -1).norm()))
{}

void SceneData::initScene()
{
    std::cout << __func__ << " - Initilizing scene..." << std::endl;

    spheres_.emplace_back(1e5, containers::Vec(1e5-9, 40.8, 81.6), containers::Vec(), containers::Vec(0.75, 0.25, 0.25), objects::Diffuse);       // Left Wall
    spheres_.emplace_back(1e5, containers::Vec(-1e5+109, 40.8, 81.6), containers::Vec(), containers::Vec(0.25, 0.25, 0.75), objects::Diffuse);     // Right Wall
    spheres_.emplace_back(1e5, containers::Vec(50, 40.8, 1e5), containers::Vec(), containers::Vec(0.75, 0.75, 0.75), objects::Diffuse);           // Back Wall
    spheres_.emplace_back(1e5, containers::Vec(50, 40.8, -1e5+175), containers::Vec(), containers::Vec(0, 0.44, 0), objects::Diffuse);            // Wall behind camera?
    spheres_.emplace_back(1e5, containers::Vec(50, 1e5, 81.6), containers::Vec(), containers::Vec(0.75, 0.75, 0.75), objects::Diffuse);           // Floor
    spheres_.emplace_back(1e5, containers::Vec(50, -1e5+81.6, 81.6), containers::Vec(), containers::Vec(0.75, 0.75, 0.75), objects::Diffuse);     // Ceiling
    spheres_.emplace_back(16.5, containers::Vec(27, 16.5, 47), containers::Vec(), containers::Vec(1, 1, 1) * 0.999, objects::Specular);           // Left Orb (Mirror like)
    spheres_.emplace_back(16.5, containers::Vec(73, 16.5, 78), containers::Vec(), containers::Vec(1, 1, 1) * 0.999, objects::Refractive);         // Right Orb (Glass ?)
    spheres_.emplace_back(600, containers::Vec(50, 681.6-.27, 81.6), containers::Vec(12, 12, 12), containers::Vec(), objects::Diffuse);           // Light source

    std::cout << __func__ << " - Done" << std::endl;
}

containers::Ray SceneData::getCamera() { return camera_; }
int SceneData::getObjectCount() { return spheres_.size(); }
objects::Sphere SceneData::getObjectAt(int id) { return spheres_.at(id); }

}  // namespace tracer::data
