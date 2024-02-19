#pragma once

#include "containers/Vec3.hpp"
#include "utils/CudaUtils.hpp"

namespace tracer::scene::objects
{

namespace
{
const double GLASS_IOR = 1.5;
const double AIR_IOR = 1.0;

__device__ containers::Vec3 calculateSpecular(const containers::Vec3& incoming, const containers::Vec3& normal)
{
    return incoming - normal * incoming.dot(normal) * 2;
}

__device__ containers::Vec3 calculateDiffuse(const containers::Vec3& normal, curandState& state)
{
    auto direction = containers::Vec3(0, 0, 0);
    while (direction == containers::Vec3(0, 0, 0))
    {
        direction = containers::Vec3(utils::one_one(state), utils::one_one(state), utils::one_one(state));
    }

    direction = direction.norm();
    return (direction.dot(normal) < 0) ? direction * -1 : direction;
}

__device__ void calculateRefreactive(containers::Vec3* refractive, const containers::Vec3& incoming,
    const containers::Vec3& normal)
{
    const auto index = AIR_IOR/GLASS_IOR;
    const auto cosIncoming = fabs(normal.dot(incoming));
    auto sinRefracted2 = pow(index, 2) * (1.0 - pow(cosIncoming, 2));

    if (sinRefracted2 > 1.0)
    {
        refractive = nullptr;
        return;
    }

    const double cosRefracted = sqrt(1.0 - sinRefracted2);
    *refractive = incoming * index + normal * (index * cosIncoming - cosRefracted);
}
}  // namespace

}  // namespace tracer::scene::objects
