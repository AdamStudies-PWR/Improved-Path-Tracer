#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils/Vec.hpp"
#include "utils/Ray.hpp"
#include "scene/Scene.hpp"

namespace tracer::renderer
{

class Renderer
{
public:
   Renderer(scene::Scene& sceneData, const int height, const int width, const int samples);

   utils::Vec* render();

private:
   bool intersect(const utils::Ray& ray, double& temp, int& id);

   const int height_;
   const int samples_;
   const int width_;
   scene::Scene& sceneData_;

   utils::Vec radiance(const utils::Ray& ray, int depth, short unsigned int* xi);
};

}  // namespace tracer::renderer
