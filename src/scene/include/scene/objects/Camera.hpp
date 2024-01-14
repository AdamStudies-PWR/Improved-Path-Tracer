#pragma once

#include "containers/Vec3.hpp"

namespace tracer::scene::objects
{

struct Camera
{
    Camera(containers::Vec3 origin=containers::Vec3(), containers::Vec3 direction=containers::Vec3(),
        containers::Vec3 orientation=containers::Vec3());

    containers::Vec3 origin_;
    containers::Vec3 direction_;
    containers::Vec3 orientation_;
};

}  // namespace tracer::scene::objects
