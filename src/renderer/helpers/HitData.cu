namespace tracer::renderer
{

struct HitData
{
    __device__ HitData(int index, double distance)
        : index_(index)
        , distance_(distance)
    {}

    int index_;
    double distance_;
};

}  // namespace tracer::renderer
