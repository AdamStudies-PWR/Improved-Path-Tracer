#pragma once

#include "scene/objects/AObject.hpp"

namespace tracer::scene::objects
{

class Plane : public AObject
{
public:
    Plane(containers::Vec3 north, containers::Vec3 east, containers::Vec3 position, containers::Vec3 emission,
        containers::Vec3 color, EReflectionType reflection);

    double intersect(const containers::Ray& ray) const override;
    containers::Ray getReflectedRay(const containers::Ray& ray, const containers::Vec3& intersection) const override;

private:
    struct PlaneEquation
    {
        PlaneEquation(bool pararelToX=false, bool pararelToY=false, bool pararelToZ=false, double aa=0.0, double bb=0.0,
            double cc=0.0);

        bool pararelToX_;
        bool pararelToY_;
        bool pararelToZ_;
        double aa_;
        double bb_;
        double cc_;
    };

    void createPlaneEquation(const containers::Vec3& topRight, const containers::Vec3& topLeft,
        const containers::Vec3& bottomRight);
    double doSimpleIntersection(const containers::Ray& ray) const;
    double doComplexIntersection(const containers::Ray& ray) const;

    PlaneEquation planeEquation_;
    // containers::Vec3 north_;
    // containers::Vec3 east_;
};

}  // namespace tracer::scene::objects
