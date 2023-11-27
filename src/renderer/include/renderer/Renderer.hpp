#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "containers/Vec.hpp"
#include "containers/Ray.hpp"
#include "scene/SceneData.hpp"

namespace tracer::renderer
{

class Renderer
{
public:
   Renderer(scene::SceneData& sceneData, const int height, const int width, const int samples);

   containers::Vec* render();

private:
   bool intersect(const containers::Ray& ray, double& temp, int& id);

   const int height_;
   const int samples_;
   const int width_;
   scene::SceneData& sceneData_;

   containers::Vec radiance(const containers::Ray& ray, int depth, short unsigned int* xi);
};

}  // namespace tracer::renderer
