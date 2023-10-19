#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils/Vec.hpp"
#include "utils/Ray.hpp"

namespace tracer::renderer
{

class Renderer
{
public:
   void initScene();
   utils::Vec* render();

private:
   utils::Vec radiance(const utils::Ray& ray, int depth, short unsigned int* xi);
};

}  // namespace tracer::renderer
