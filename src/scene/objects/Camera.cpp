#include "scene/objects/Camera.hpp"

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}  // namespace

Camera::Camera(Vec3 origin, Vec3 direction, Vec3 orientation)
    : origin_(origin)
    , direction_(direction)
    , orientation_(orientation)
{}

}  // namespace tracer::scene::objects
