#include "Plane.hpp"

#include <iostream>

namespace tracer::scene::objects
{

namespace
{
using namespace containers;
}

Plane::Plane(Vec3 north, Vec3 east, Vec3 position, Vec3 emission, Vec3 color, EReflectionType reflection)
    : AObject(position, emission, color, reflection)
{
    const auto topRight = position_ + north + east;
    const auto topLeft = position_ + north + (east * -1);
    const auto bottomRight = position_ + (north * -1) + east;
    // const auto bottomLeft = position_ + (north * -1) + (east * -1);

    createPlaneEquation(topRight, topLeft, bottomRight);
}

double Plane::intersect(const Ray& ray) const
{
    double intersection = 0.0;

    if (planeEquation_.pararelToX_ or planeEquation_.pararelToY_ or planeEquation_.pararelToZ_)
    {
        intersection = doSimpleIntersection(ray);
    }
    else
    {
        intersection = doComplexIntersection(ray);
    }

    // TODO: Bounding straights

    return (intersection <= 0.0) ? 0.0 : intersection;
}

Ray Plane::getReflectedRay(const Ray&, const Vec3&) const
{
    return Ray();
}

Plane::PlaneEquation::PlaneEquation(bool pararelToX, bool pararelToY, bool pararelToZ, double aa, double bb, double cc)
    : pararelToX_(pararelToX)
    , pararelToY_(pararelToY)
    , pararelToZ_(pararelToZ)
    , aa_(aa)
    , bb_(bb)
    , cc_(cc)
{}

void Plane::createPlaneEquation(const Vec3& topRight, const Vec3& topLeft, const Vec3& bottomRight)
{
    if (topRight.xx_ == topLeft.xx_ and topRight.xx_ == bottomRight.xx_)
    {
        planeEquation_ = PlaneEquation(true, false, false, topRight.xx_, 0.0, 0.0);
        return;
    }

    if (topRight.yy_ == topLeft.yy_ and topRight.yy_ == bottomRight.yy_)
    {
        planeEquation_ = PlaneEquation(false, true, false, 0.0, topRight.yy_, 0.0);
        return;
    }

    if (topRight.zz_ == topLeft.zz_ and topRight.zz_ == bottomRight.zz_)
    {
        planeEquation_ = PlaneEquation(false, false, true, 0.0, 0.0, topRight.zz_);
        return;
    }

    const auto kk = topRight.xx_*topLeft.yy_ - topLeft.xx_*topRight.yy_;
    const auto ii = topRight.xx_*topLeft.zz_ - topLeft.xx_*topRight.zz_;
    const auto jj = topRight.xx_ - topLeft.xx_;
    const auto mm = topRight.xx_*bottomRight.zz_*kk - ii*topRight.xx_*bottomRight.yy_ - kk*topRight.zz_
        - topRight.yy_*bottomRight.xx_*ii;
    const auto nn = topRight.yy_ * bottomRight.xx_ * jj - kk - jj * topRight.xx_ * bottomRight.yy_ + kk * topRight.xx_;

    const auto aa = (-topRight.yy_ * ((nn/mm * ii - jj)/kk) + nn/mm * topRight.zz_ - 1)/topRight.xx_;
    const auto bb = (nn/mm * ii - jj)/kk;
    planeEquation_ = PlaneEquation(false, false, false, aa, bb, -nn/mm);
}

double Plane::doSimpleIntersection(const Ray& ray) const
{
    if (planeEquation_.pararelToX_)
    {
        return planeEquation_.aa_ / ray.direction_.xx_;
    }
    else if (planeEquation_.pararelToY_)
    {
        return planeEquation_.bb_ / ray.direction_.yy_;
    }
    else
    {
        return planeEquation_.cc_ / ray.direction_.zz_;
    }
}

double Plane::doComplexIntersection(const Ray& ray) const
{
    const auto temp = planeEquation_.aa_ * ray.direction_.xx_ + planeEquation_.bb_ * ray.direction_.yy_
        + planeEquation_.cc_ * ray.direction_.zz_;

    return -1/temp;
}

}  // namespace tracer::scene::objects
