#include "PixelData.hpp"

namespace tracer::renderer
{

namespace
{
using namespace containers;
}

PixelData::PixelData(const double stepX, const double stepZ, const containers::Vec3 gaze)
    : stepX_(stepX)
    , stepZ_(stepZ)
    , gaze_(gaze)
{}

}  // namespace tracer::renderer
