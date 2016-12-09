// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring> // memset
#include <memory>
#include <random>
#include <cassert>
#include <algorithm>

#include <GL/glew.h>

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/detail/platform.h>

#include <visionaray/texture/texture.h>

#include <visionaray/aligned_vector.h>
#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h> // for make_kernel_params(...)
#include <visionaray/material.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/timer.h>
#include <common/viewer_glut.h>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

#include "basic_quad.h"
#include "snex.h"

using namespace visionaray;

using viewer_type = viewer_glut;


template <typename T>
basic_quad<T> make_quad(
        vector<3, T> v1,
        vector<3, T> v2,
        vector<3, T> v3,
        vector<3, T> v4
        )
{
    basic_quad<T> q;
    q.v1 = v1;
    q.v2 = v2;
    q.v3 = v3;
    q.v4 = v4;
    q.prim_id = 0;
    q.geom_id = 0;
    return q;
}


template <typename S>
struct benchmark_impl
{
    template <typename V1, typename V2>
    void pack_rays(V1& rays_cpu, V2 const& rays)
    {
        assert(rays.size() % 8 == 0);

        const size_t packet_size = simd::num_elements<S>::value;

        for (size_t i=0; i<rays.size()/packet_size; i++)
        {
            std::array<basic_ray<float>, packet_size> ra;

            for (size_t e=0; e<packet_size; ++e)
            {
                ra[e] = rays[i*packet_size + e];
            }

            rays_cpu.push_back(simd::pack(ra));
        }
    }
};

template <>
struct benchmark_impl <float>
{
    template <typename V1, typename V2>
    void pack_rays(V1& rays_cpu, V2 const& rays)
    {
        rays_cpu.resize(rays.size());
        std::copy(rays.cbegin(), rays.cend(), rays_cpu.begin());
    }
};

#ifdef __CUDACC__
template
<typename Intersector, typename quad_type, typename ray_type>
__global__ void cuda_kernel(ray_type *rays, unsigned int ray_count, quad_type *first, quad_type *last)
{
    Intersector i;
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < ray_count)
    {
        ray_type r = rays[index];
        volatile auto hr = closest_hit(r, first, last, i);
    }
}
#endif

template <typename S>
struct benchmark
{
    typedef basic_quad<float> quad_type;
    typedef quad_prim<float> quad_type_opt;
    typedef basic_ray<float> ray_type;
    typedef basic_ray<S> ray_type_cpu;

    aligned_vector<quad_type_opt> quads_opt;
    aligned_vector<quad_type, 32> quads;
    aligned_vector<ray_type, 32> rays;

    aligned_vector<basic_ray<S>, 32> rays_cpu;

    std::string name_;
    bool cuda_test;
    int cuda_block_size = 192;


    const unsigned int quad_count = 100000;
    const unsigned int ray_count = (1<<18);


    typedef std::default_random_engine rand_engine;
    typedef std::uniform_real_distribution<float> uniform_dist;

    rand_engine  rng;
    uniform_dist dist;

    benchmark(std::string name, bool cuda_test)
        : name_(name)
        , rng(0)
        , dist(0, 1)
        , cuda_test(cuda_test)
    {
        generate_quads();
        generate_rays();
    }

    void generate_quads()
    {
        for (size_t i=0; i<quad_count; i++)
        {
            vec3 u;
            vec3 v;
            vec3 w = normalize(vec3(dist(rng), dist(rng), dist(rng)));
            make_orthonormal_basis(u, v, w);

            //vec3 center = vec3(vec3(dist(rng), dist(rng), dist(rng)));
            vec3 center = vec3(0.f);

            quad_type quad = make_quad(
                    u * dist(rng) + center,
                    v * dist(rng) + center,
                    -u * dist(rng) + center,
                    -v * dist(rng) + center
                    );
            quads.push_back(quad);


            quads_opt.push_back(quad_type_opt::make_quad(
                    u * dist(rng) + center,
                    v * dist(rng) + center,
                    -u * dist(rng) + center,
                    -v * dist(rng) + center
                    ));
        }
    }

    void generate_rays()
    {
        for (size_t i=0; i<ray_count; i++)
        {
            ray r;

            vec3 origin(0.f, 0.f, 4.f);
            vec3 dir = normalize(vec3(dist(rng), dist(rng), 0.f) - origin);

            r.ori = origin;
            r.dir = dir;

            rays.push_back(r);
        }
    }

    void init()
    {
        benchmark_impl<S> impl;
        impl.pack_rays(rays_cpu, rays);

#ifdef __CUDACC__
        init_device_data();
#endif
    }


    double operator()()
    {
        if (!cuda_test)
        {
            if (name_ == "mt bl uv")
                return run_test<quad_intersector_mt_bl_uv>();
            if (name_ == "pluecker")
                return run_test<quad_intersector_pluecker>();
            if (name_ == "project 2d")
                return run_test<quad_intersector_project_2D>();
            if (name_ == "uv")
                return run_test<quad_intersector_uv>();
        }
        else
        {
#ifdef __CUDACC__
            return run_cuda_test();
#endif
        }
        return 0.0;
    }

    template <typename intersector>
    double run_test()
    {
        intersector i;

        timer t;

        for (auto &r: rays_cpu)
        {
            volatile auto hr = closest_hit(r, quads.begin(), quads.end(), i);
        }

        return t.elapsed();
    }

#ifdef __CUDACC__
    thrust::device_vector<quad_type> d_quads;
    thrust::device_vector<quad_type_opt> d_quads_opt;
    thrust::device_vector<ray_type> d_rays;

    void init_device_data()
    {
        d_quads = thrust::device_vector<quad_type>(quads);
        d_rays = thrust::device_vector<ray_type>(rays);
        d_quads_opt = thrust::device_vector<quad_type_opt>(quads_opt);
    }

    double run_cuda_test()
    {
        dim3 block_size(cuda_block_size);
        dim3 grid_size(div_up(ray_count, block_size.x));

        if (name_ == "opt")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_opt> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads_opt.data()),
                    thrust::raw_pointer_cast(d_quads_opt.data()) + d_quads_opt.size()
                    );

            return t.elapsed();
        }
        else if (name_ == "mt bl uv")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_mt_bl_uv> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size()
                    );

            return t.elapsed();
        }
        else if (name_ == "pluecker")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_pluecker> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size()
                    );

            return t.elapsed();
        }
        else if (name_ == "project 2d")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_project_2D> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size()
                    );

            return t.elapsed();
        }
        else if (name_ == "uv")
        {
            cuda::timer t;

            cuda_kernel<quad_intersector_uv> <<<grid_size, block_size>>> (
                    thrust::raw_pointer_cast(d_rays.data()),
                    ray_count,
                    thrust::raw_pointer_cast(d_quads.data()),
                    thrust::raw_pointer_cast(d_quads.data()) + d_quads.size()
                    );

            return t.elapsed();
        }

        return 0.0;
    }
#endif
};


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    int dry_runs = 3;   // some dry runs to fill the caches
    int bench_runs = 10;
    int bs = 192;

    int do_cuda_test = 0;

    std::string name = "opt";


    using namespace support;

    cl::CmdLine cmd;

    auto bsref = cl::makeOption<int&>(
            cl::Parser<>(), cmd, "blocksize",
            cl::ArgName("blocksize"),
            cl::ArgRequired,
            cl::init(bs),
            cl::Desc("CUDA block size")
            );

    auto nref = cl::makeOption<std::string&>(
            cl::Parser<>(), cmd, "intersect",
            cl::ArgName("intersect"),
            cl::ArgRequired,
            cl::init(name),
            cl::Desc("Intersection algorithm")
            );

    auto ctref = cl::makeOption<int&>(
            cl::Parser<>(), cmd, "cuda_test",
            cl::ArgName("cuda_test"),
            cl::init(do_cuda_test),
            cl::Desc("Whether to test cuda algorithm")
            );

    try
    {
        auto args = std::vector<std::string>(argv + 1, argv + argc);

        cl::expandWildcards(args);
        //cl::expandResponseFiles(args, cl::TokenizeWindows());
        cl::expandResponseFiles(args, cl::TokenizeUnix());

        cmd.parse(args);
    }
    catch (std::exception& e)
    {
        std::cout << "error: " << e.what() << '\n';
        std::cout << '\n';
        std::cout << cmd.help("benchmark") << '\n';
        return -1;
    }

    //benchmark<simd::float8> b;
    benchmark<float> b(name, do_cuda_test);
    b.init();
    b.cuda_block_size = bs;

    for (int i = 0; i < dry_runs; ++i)
    {
        volatile double t = b() * 1000.0;
    }

    std::vector<double> times(bench_runs);
    for (int i = 0; i < bench_runs; ++i)
    {
        times[i] = b() * 1000.0;
    }

    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    std::cout << "Benchmark:   " << name << '\n';
    if (do_cuda_test)
    std::cout << "CUDA grid:   " << div_up((int)b.rays.size(), b.cuda_block_size) << " blocks of size " << b.cuda_block_size << '\n';
    std::cout << "Num rays:    " << b.rays.size() << '\n';
    std::cout << "Rays/sec:    " << b.rays.size() * bench_runs * 1000.0 / sum << '\n';
    std::cout << "Average:     " << sum / bench_runs << " ms\n";
    std::cout << "Median:      " << times[bench_runs / 2] << " ms\n";
    std::cout << "Max:         " << times.back() << " ms\n";
    std::cout << "Min:         " << times[0] << " ms\n";

    return 0;
}
