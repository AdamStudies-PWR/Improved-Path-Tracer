#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <iostream>
#include <vector>

struct Vec
{
    double xx_;
    double yy_;
    double zz_;

    Vec(double xx=0, double yy=0, double zz=0)
        : xx_(xx)
        , yy_(yy)
        , zz_(zz)
    {}

    Vec operator* (double number) const { return Vec(xx_ * number, yy_ * number, zz_ * number); }
    Vec operator+ (const Vec& summand) const { return Vec(xx_ + summand.xx_, yy_ + summand.yy_, zz_ + summand.zz_); }

    Vec operator% (Vec& input)
    {
        return Vec(yy_*input.zz_ - zz_*input.yy_, zz_*input.xx_ - xx_*input.zz_, xx_*input.yy_ - yy_*input.xx_);
    }

    Vec operator- (const Vec& subtrahend) const
    {
        return Vec(xx_ - subtrahend.xx_, yy_ - subtrahend.yy_, zz_ - subtrahend.zz_);
    }

    double dot(const Vec& input) const
    {
        return xx_*input.xx_ + yy_*input.yy_ + zz_*input.zz_;
    }

    Vec& norm()
    {
        return *this = *this * (1/sqrt(xx_*xx_ + yy_*yy_ + zz_*zz_));
    }

    Vec mult(const Vec& input) const
    {
        return Vec(xx_*input.xx_, yy_*input.yy_, zz_*input.zz_);
    }
};

struct Ray
{
    Vec oo_;
    Vec dd_;

    Ray(Vec oo, Vec dd)
        : oo_(oo)
        , dd_(dd)
    {}
};

enum EReflectionType
{
    Diffuse,
    Specular,
    Refractive
};

struct Sphere
{
    double radius_;
    Vec position_;
    Vec emission_;
    Vec color_;
    EReflectionType relfection_;

    Sphere(double radius, Vec position, Vec emission, Vec color, EReflectionType relfection)
        : radius_(radius)
        , position_(position)
        , emission_(emission)
        , color_(color)
        , relfection_(relfection)
    {}

    double intersect(const Ray& ray) const
    {
        Vec op = position_ - ray.oo_;
        double temp;
        double eps = 1e-4;
        double b = op.dot(ray.dd_);
        double det = b*b - op.dot(op) + radius_*radius_;

        return (temp = b - det) > eps ? temp : ((temp = b + det) > eps ? temp : 0);
    }
};

namespace
{

int HEIGHT = 768;
int WIDTH = 1024;
int samps = 1250;
Ray camera(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm());

std::vector<Sphere> spheres;

inline bool intersect(const Ray& ray, double& temp, int& id)
{
    double n = sizeof(spheres)/sizeof(Sphere);
    double dd;
    double inf=temp=1e20;

    for(int i = int(n); i--;)
    {
        if ((dd = spheres[i].intersect(ray)) && dd < temp)
        {
            temp = dd;
            id = i;
        }
    }

    return temp < inf;
}

inline double clamp(double x)
{
    return x<0 ? 0 : x>1 ? 1 : x;
}

inline int toInt(double x)
{
    return int(pow(clamp(x),1/2.2)*255+.5);
}

}  // namespace

void initScene()
{
    std::cout << __func__ << " - Initilizing scene..." << std::endl;

    spheres.emplace_back(1e5, Vec(1e5+1, 40.8, 81.6), Vec(), Vec(0.75, 0.25, 0.25), Diffuse);       // Left
    spheres.emplace_back(1e5, Vec(-1e5+99, 40.8, 81.6), Vec(), Vec(0.25, 0.25, 0.75), Diffuse);     // Right
    spheres.emplace_back(1e5, Vec(50, 40.8, 1e5), Vec(), Vec(0.75, 0.75, 0.75), Diffuse);           // Back
    spheres.emplace_back(1e5, Vec(50, 40.8, -1e5+170), Vec(), Vec(), Diffuse);                      // Front
    spheres.emplace_back(1e5, Vec(50, 1e5, 81.6), Vec(), Vec(0.75, 0.75, 0.75), Diffuse);           // Bottom
    spheres.emplace_back(1e5, Vec(50, -1e5+81.6, 81.6), Vec(), Vec(0.75, 0.75, 0.75), Diffuse);     // Top
    spheres.emplace_back(16.5, Vec(27, 16.5, 47), Vec(), Vec(1, 1, 1) * 0.999, Specular);           // Mirror
    spheres.emplace_back(16.5, Vec(73, 16.5, 78), Vec(), Vec(1, 1, 1) * 0.999, Refractive);         // Glass
    spheres.emplace_back(600, Vec(50, 681.6-.27, 81.6), Vec(12, 12, 12), Vec(), Diffuse);           // Lite (?)

    std::cout << __func__ << " - Done" << std::endl;
}

Vec radiance(const Ray& ray, int depth, uint16_t* xi)
{
    double temp;
    int id = 0;

    if (!intersect(ray, temp, id))
    {
        return Vec();
    }

    const Sphere& sphere = spheres[id];

    Vec xx = ray.oo_ + ray.dd_ * temp;
    Vec nn = (xx - sphere.position_).norm();
    Vec nl = nn.dot(ray.dd_) < 0 ? nn : nn * -1;
    Vec ff = sphere.color_;

    double pp = ff.xx_ > ff.yy_ && ff.xx_ > ff.zz_ ? ff.xx_ : ff.yy_ > ff.zz_ ? ff.yy_ : ff.zz_;
    if (++depth > 5)
    {
        if (erand48(xi) < pp)
        {
            ff = ff * (1/pp);
        }
        else return sphere.emission_;
    }

    if (sphere.relfection_ == Diffuse)
    {
        double r1 = 2*M_PI*erand48(xi);
        double r2 = erand48(xi);
        double r2s = sqrt(r2);

        Vec ww = nl;
        Vec uu = ((fabs(ww.xx_) > 0.1 ? Vec(0, 1, 0) : Vec(1, 0, 0))%ww).norm();
        Vec vv = ww%uu;
        Vec dd = (uu*cos(r1)*r2s + vv*sin(r1)*r2s + ww*sqrt(1-r2)).norm();

        return sphere.emission_ + ff.mult(radiance(Ray(xx, dd), depth, xi));
    }
    else if (sphere.relfection_ == Specular)
    {
        return sphere.emission_ + ff.mult(radiance(Ray(xx, ray.dd_ - nn*2*nn.dot(ray.dd_)), depth, xi));
    }

    Ray reflectedRay(xx, ray.dd_ - nn*2*nn.dot(ray.dd_));
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

    Vec tdir = (ray.dd_*nnt - nn*((into ? 1 : -1) * (ddn*nnt+sqrt(cos2t)))).norm();
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
        radiance(reflectedRay, depth, xi)*RP : radiance(Ray(xx, tdir), depth, xi)* TP) :
        radiance(reflectedRay, depth, xi)*Re + radiance(Ray(xx, tdir), depth, xi)*Tr);
}

// refactor this !!!
Vec* render()
{
    Vec cx = Vec(WIDTH*0.5135/HEIGHT);
    Vec cy = (cx%camera.dd_).norm()*0.5135;
    Vec r;
    Vec* c = new Vec[WIDTH*HEIGHT];

    #pragma omp parallel for schedule(dynamic, 1) private(r)
    for (int y=0; y<HEIGHT; y++)
    {
        fprintf(stderr,"\rRendering (%d spp) %5.2f%%",samps*4,100.*y/(HEIGHT-1));
        for (unsigned short x=0, Xi[3]={0,0,y*y*y}; x<WIDTH; x++)   // Loop cols
        {
            for (int sy=0, i=(HEIGHT-y-1)*WIDTH+x; sy<2; sy++)     // 2x2 subpixel rows
            {
                for (int sx=0; sx<2; sx++, r=Vec())
                {        // 2x2 subpixel cols
                    for (int s=0; s<samps; s++)
                    {
                        double r1=2*erand48(Xi), dx=r1<1 ? sqrt(r1)-1: 1-sqrt(2-r1);
                        double r2=2*erand48(Xi), dy=r2<1 ? sqrt(r2)-1: 1-sqrt(2-r2);
                        Vec d = cx*( ( (sx+.5 + dx)/2 + x)/WIDTH - .5) +
                                cy*( ( (sy+.5 + dy)/2 + y)/HEIGHT - .5) + camera.dd_;
                        r = r + radiance(Ray(camera.oo_+d*140,d.norm()), 0, Xi)*(1./samps);
                    } // Camera rays are pushed ^^^^^ forward to start in interior
                c[i] = c[i] + Vec(clamp(r.xx_), clamp(r.yy_), clamp(r.zz_))*0.25;
                }
            }
        }
    }

    return c;
}

Vec* measure()
{
    std::cout << __func__ << " - Begining render..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto* image = render();
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << __func__ << " - Done" << std::endl;
    std::cout << __func__ << " - Render took: "
        << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << " microseconds" << std::endl;

    return image;
}

// Refactor this
void saveImage(Vec* image)
{
    std::cout << __func__ << " - saving render..." << std::endl;

    FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
    fprintf(f, "P3\n%d %d\n%d\n", WIDTH, HEIGHT, 255);
    for (int i=0; i<WIDTH*HEIGHT; i++)
        fprintf(f,"%d %d %d ", toInt(image[i].xx_), toInt(image[i].yy_), toInt(image[i].zz_));

    std::cout << __func__ << " - Done" << std::endl;
}

int main(int argc, char *argv[])
{
    initScene();
    auto* image = measure();
    saveImage(image);

    return 0;
}
