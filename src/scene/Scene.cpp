#include "scene/Scene.hpp"

#include <iostream>

namespace tracer::scene
{

Scene::Scene()
    : camera_(utils::Ray(utils::Vec(50, 52, 295.6), utils::Vec(0, -0.042612, -1).norm()))
{}

void Scene::initScene()
{
    std::cout << __func__ << " - Initilizing scene..." << std::endl;

    spheres_.emplace_back(1e5, utils::Vec(1e5+1, 40.8, 81.6), utils::Vec(), utils::Vec(0.75, 0.25, 0.25), objects::Diffuse);       // Left Wall
    spheres_.emplace_back(1e5, utils::Vec(-1e5+99, 40.8, 81.6), utils::Vec(), utils::Vec(0.25, 0.25, 0.75), objects::Diffuse);     // Right Wall
    spheres_.emplace_back(1e5, utils::Vec(50, 40.8, 1e5), utils::Vec(), utils::Vec(0.75, 0.75, 0.75), objects::Diffuse);           // Back Wall
    spheres_.emplace_back(1e5, utils::Vec(50, 40.8, -1e5+170), utils::Vec(), utils::Vec(0, 0.44, 0), objects::Diffuse);            // Wall behind camera?
    spheres_.emplace_back(1e5, utils::Vec(50, 1e5, 81.6), utils::Vec(), utils::Vec(0.75, 0.75, 0.75), objects::Diffuse);           // Floor
    spheres_.emplace_back(1e5, utils::Vec(50, -1e5+81.6, 81.6), utils::Vec(), utils::Vec(0.75, 0.75, 0.75), objects::Diffuse);     // Ceiling
    spheres_.emplace_back(16.5, utils::Vec(27, 16.5, 47), utils::Vec(), utils::Vec(1, 1, 1) * 0.999, objects::Specular);           // Left Orb (Mirror like)
    spheres_.emplace_back(16.5, utils::Vec(73, 16.5, 78), utils::Vec(), utils::Vec(1, 1, 1) * 0.999, objects::Refractive);         // Right Orb (Glass ?)
    spheres_.emplace_back(600, utils::Vec(50, 681.6-.27, 81.6), utils::Vec(12, 12, 12), utils::Vec(), objects::Diffuse);           // Light source

    std::cout << __func__ << " - Done" << std::endl;
}

utils::Ray Scene::getCamera() { return camera_; }
int Scene::getObjectCount() { return spheres_.size(); }
objects::Sphere Scene::getObjectAt(int id) { return spheres_.at(id); }

}  // namespace tracer::scene
