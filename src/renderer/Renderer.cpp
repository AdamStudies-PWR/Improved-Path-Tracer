#include "renderer/Renderer.hpp"

#include <vector>
#include <iostream>

#include "objects/Sphere.hpp"

namespace tracer::renderer
{

namespace
{
inline double clamp(double x)
{
    return x < 0 ? 0 : x > 1 ? 1 : x;
}
}  // namespace

Renderer::Renderer(scene::Scene& sceneData, const int height, const int width, const int samples)
    : height_(height)
    , samples_(samples)
    , width_(width)
    , sceneData_(sceneData)
{}

utils::Vec Renderer::radiance(const utils::Ray& ray, int depth, short unsigned int* xi)
{
    double temp;
    int id = 0;

    if (!intersect(ray, temp, id))
    {
        return utils::Vec();
    }

    const objects::Sphere& sphere = sceneData_.getObjectAt(id);

    utils::Vec xx = ray.oo_ + ray.dd_ * temp;
    utils::Vec nn = (xx - sphere.position_).norm();
    utils::Vec nl = nn.dot(ray.dd_) < 0 ? nn : nn * -1;
    utils::Vec ff = sphere.color_;

    double pp = ff.xx_ > ff.yy_ && ff.xx_ > ff.zz_ ? ff.xx_ : ff.yy_ > ff.zz_ ? ff.yy_ : ff.zz_;
    if (++depth > 5)
    {
        if (erand48(xi) < pp)
        {
            ff = ff * (1/pp);
        }
        else return sphere.emission_;
    }

    if (sphere.relfection_ == objects::Diffuse)
    {
        double r1 = 2*M_PI*erand48(xi);
        double r2 = erand48(xi);
        double r2s = sqrt(r2);

        utils::Vec ww = nl;
        utils::Vec uu = ((fabs(ww.xx_) > 0.1 ? utils::Vec(0, 1, 0) : utils::Vec(1, 0, 0))%ww).norm();
        utils::Vec vv = ww%uu;
        utils::Vec dd = (uu*cos(r1)*r2s + vv*sin(r1)*r2s + ww*sqrt(1-r2)).norm();

        return sphere.emission_ + ff.mult(radiance(utils::Ray(xx, dd), depth, xi));
    }
    else if (sphere.relfection_ == objects::Specular)
    {
        return sphere.emission_ + ff.mult(radiance(utils::Ray(xx, ray.dd_ - nn*2*nn.dot(ray.dd_)), depth, xi));
    }

    utils::Ray reflectedRay(xx, ray.dd_ - nn*2*nn.dot(ray.dd_));
    bool into = nn.dot(nl) > 0;
    double nc = 1;
    double nt = 1.5;
    double nnt = into ? nc/nt : nt/nc;
    double ddn = ray.dd_.dot(nl);
    double cos2t;

    if ((cos2t=1-nnt*nnt*(1-ddn*ddn)) < 0)
    {
        return sphere.emission_ + ff.mult(radiance(reflectedRay, depth, xi));
    }

    utils::Vec tdir = (ray.dd_*nnt - nn*((into ? 1 : -1) * (ddn*nnt+sqrt(cos2t)))).norm();
    double aa = nt - nc;
    double bb = nt + nc;
    double R0 = aa*aa/(bb*bb);
    double cc = 1 - (into ? -ddn : tdir.dot(nn));
    double Re = R0 + (1 - R0)*cc*cc*cc*cc*cc;
    double Tr = 1 - Re;
    double PP = 0.25 + 0.5 * Re;
    double RP = Re/PP;
    double TP = Tr/(1-PP);

    return sphere.emission_ + ff.mult(depth > 2 ? (erand48(xi) < PP ?
        radiance(reflectedRay, depth, xi)*RP : radiance(utils::Ray(xx, tdir), depth, xi)* TP) :
        radiance(reflectedRay, depth, xi)*Re + radiance(utils::Ray(xx, tdir), depth, xi)*Tr);
}

// refactor this !!!
utils::Vec* Renderer::render()
{
    auto camera = sceneData_.getCamera();
    utils::Vec cx = utils::Vec(width_*0.5135/height_);
    utils::Vec cy = (cx%camera.dd_).norm()*0.5135;
    utils::Vec r;
    utils::Vec* c = new utils::Vec[width_*height_];

#pragma omp parallel for schedule(dynamic, 1) private(r)
    for (int y=0; y<height_; y++)
    {
        fprintf(stderr,"\rRendering (%d spp) %5.2f%%",samples_*4,100.*y/(height_-1));
        for (unsigned short x=0, Xi[3]={0,0,y*y*y}; x<width_; x++)   // Loop cols
        {
            for (int sy=0, i=(height_-y-1)*width_+x; sy<2; sy++)     // 2x2 subpixel rows
            {
                for (int sx=0; sx<2; sx++, r=utils::Vec())
                {        // 2x2 subpixel cols
                    for (int s=0; s<samples_; s++)
                    {
                        double r1=2*erand48(Xi), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
                        double r2=2*erand48(Xi), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
                        utils::Vec d = cx*( ( (sx+.5 + dx)/2 + x)/width_ - .5) +
                                cy*( ( (sy+.5 + dy)/2 + y)/height_ - .5) + camera.dd_;
                        r = r + radiance(utils::Ray(camera.oo_+d*140,d.norm()), 0, Xi)*(1./samples_);
                    } // Camera rays are pushed ^^^^^ forward to start in interior
                c[i] = c[i] + utils::Vec(clamp(r.xx_), clamp(r.yy_), clamp(r.zz_))*0.25;
                }
            }
        }
    }

    return c;
}

bool Renderer::intersect(const utils::Ray& ray, double& temp, int& id)
{
    const auto n = sceneData_.getObjectCount();
    double dd;
    double inf = temp = 1e20;

    for(int i = n; i--;)
    {
        if ((dd = sceneData_.getObjectAt(i).intersect(ray)) && dd < temp)
        {
            temp = dd;
            id = i;
        }
    }

    return temp < inf;
}

}  // namespace tracer::renderer
