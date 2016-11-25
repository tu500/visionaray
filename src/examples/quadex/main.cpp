// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring> // memset
#include <memory>
#include <random>
#include <cassert>
#include <algorithm>

#include <GL/glew.h>

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

#include <thrust/device_vector.h>

#include "quad.h"

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<simd::float8>;

    renderer()
        : viewer_type(512, 512, "Visionaray Quad Example")
        , bbox({ -20.0f, -20.0f, -20.0f }, { 20.0f, 20.0f, 20.0f })
//      , host_sched(1)
        , host_sched(8)
    {
        quad.v1 = vec3(-15.0f,  0.0f,  15.0f);
        quad.v2 = vec3( 15.0f,  0.0f,  15.0f);
        quad.v3 = vec3( 15.0f,  0.0f, -15.0f);
        quad.v4 = vec3(-15.0f,  0.0f, -29.0f);
        quad.prim_id = 0;
        quad.geom_id = 0;

        material.set_ca( from_rgb(0.2f, 0.2f, 0.2f) );
        material.set_cd( from_rgb(0.8f, 0.8f, 0.8f) );
        material.set_cs( from_rgb(0.1f, 0.1f, 0.1f) );
        material.set_ka( 1.0f );
        material.set_kd( 1.0f );
        material.set_ks( 1.0f );
        material.set_specular_exp( 32.0f );

        vec3 chess[4] = {
            vec3(1.0f, 0.0f, 0.0f),
            vec3(1.0f, 1.0f, 1.0f),
            vec3(1.0f, 0.0f, 0.0f),
            vec3(1.0f, 1.0f, 1.0f)
            };

        tex = texture<vec3, 2>(2, 2);
        tex.set_address_mode( Wrap );
        tex.set_filter_mode( Nearest );
        tex.set_data(chess);

        tex_coords.resize(4);
        tex_coords[0] = vec2(0.0f, 0.0f);
        tex_coords[1] = vec2(1.0f, 0.0f);
        tex_coords[2] = vec2(1.0f, 1.0f);
        tex_coords[3] = vec2(0.0f, 1.0f);
    }

    aabb                                        bbox;
    camera                                      cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;


    // rendering data

    basic_quad<float>                           quad;
    plastic<float>                              material;
    texture<vec3, 2>                            tex;
    aligned_vector<vec2>                        tex_coords;

protected:

    void on_display();
    void on_resize(int w, int h);

};


//-------------------------------------------------------------------------------------------------
// Display function
//

void renderer::on_display()
{
    // some setup
    auto sparams = make_sched_params(
            cam,
            host_rt
            );


    aligned_vector<basic_quad<float>> primitives;
    primitives.push_back(quad);

    aligned_vector<plastic<float>> materials;
    materials.push_back(material);

    aligned_vector<vec3> normals;
    normals.push_back(vec3(0, 1, 0));


    // headlight
    point_light<float> light;
    light.set_cl( vec3(1.0f, 1.0f, 1.0f) );
    light.set_kl( 1.0f );
    light.set_position(cam.eye());

    aligned_vector<point_light<float>> lights;
    lights.push_back(light);


    auto kparams = make_kernel_params(
            normals_per_face_binding{},
            primitives.data(),
            primitives.data() + primitives.size(),
            normals.data(),
            tex_coords.data(),
            materials.data(),
            &tex,
            lights.data(),
            lights.data() + lights.size(),
            1,          // num bounces - irrelevant for primary ray shading
            1E-3f       // a tiny number - also irrelevant for primary ray shading
            );


    simple::kernel<decltype(kparams)> kern;
    kern.params = kparams;

    timer t;

    host_sched.frame(kern, sparams);

    std::cout << "Elapsed rendering time: " << t.elapsed() << '\n';


    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();
}


//-------------------------------------------------------------------------------------------------
// resize event
//

void renderer::on_resize(int w, int h)
{
    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    host_rt.resize(w, h);

    viewer_type::on_resize(w, h);
}


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

template <typename S>
struct benchmark
{
    typedef basic_quad<float> quad_type;
    typedef basic_ray<float> ray_type;
    typedef basic_ray<S> ray_type_cpu;

    aligned_vector<quad_type, 32> quads;
    aligned_vector<ray_type, 32> rays;

    aligned_vector<basic_ray<S>, 32> rays_cpu;



    const size_t quad_count = 1000;
    const size_t ray_count = (1<<20);


    typedef std::default_random_engine rand_engine;
    typedef std::uniform_real_distribution<float> uniform_dist;

    rand_engine  rng;
    uniform_dist dist;

    benchmark()
        : rng(0), dist(0, 1)
    {
        generate_quads();
        generate_rays();
    }

    void generate_quads()
    {
        for (size_t i=0; i<quad_count; i++)
        {
            vec3 normal = normalize(vec3(dist(rng), dist(rng), dist(rng)));

            auto w = normal;
            auto v = select(
                    abs(w.x) > abs(w.y),
                    normalize( vec3(-w.z, 0.f, w.x) ),
                    normalize( vec3(0.f, w.z, -w.y) )
                    );
            auto u = cross(v, w);

            //vec3 center = vec3(vec3(dist(rng), dist(rng), dist(rng)));
            vec3 center = vec3(0.f);

            quad_type quad = make_quad(
                    u * dist(rng) + center,
                    v * dist(rng) + center,
                    -u * dist(rng) + center,
                    -v * dist(rng) + center
                    );
            quads.push_back(quad);
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
    }


    void operator() ()
    {
        run_test<quad_intersector_mt_bl_uv>("mt bl uv");
        run_test<quad_intersector_pluecker>("pluecker");
        run_test<quad_intersector_project_2D>("project 2d");
        run_test<quad_intersector_uv>("uv");
    }

    template <typename intersector>
    void run_test(std::string name)
    {
        intersector i;

        timer t;

        for (auto &r: rays_cpu)
        {
            volatile auto hr = closest_hit(r, quads.begin(), quads.end(), i);
        }

        volatile auto elapsed = t.elapsed();

        std::cout << name << " elapsed time: " << elapsed << '\n';
    }

#ifdef __CUDACC__
    thrust::device_vector<quad_type> d_quads;
    thrust::device_vector<ray_type> d_rays;

    void init_device_data()
    {
        d_quads = thrust::device_vector<quad_type>(quads);
        d_rays = thrust::device_vector<quad_type>(rays);
    }

    void run_cuda_test()
    {
        timer t;

        dim3 block_size(32);
        dim3 grid_size(
                div_up(ray_count, block_size.x)
                );

        cuda_kernel <<<grid_size, block_size>>> (
                thrust::raw_pointer_cast(d_rays.data()),
                thrust::raw_pointer_cast(d_quads.data()),
                thrust::raw_pointer_cast(d_quads.data()) + d_quads.size()
                );

        volatile auto elapsed = t.elapsed();
        std::cout << name << " elapsed time: " << elapsed << '\n';
    }

    template
    <typename Intersector>
    __global__ void cuda_kernel(ray_type *rays, quad_type *first, quad_type *last)
    {
        Intersector i;
        auto index = blockIdx.x * blockSize.x + threadIdx.x;
        if (index < ray_count)
        {
            ray_type r = rays[index];
            volatile auto hr = closest_hit(r, first, last, i);
        }
    }
#endif
};


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
    benchmark<simd::float8> b;
    b.init();
    b();
    return 0;

    renderer rend;

    try
    {
        rend.init(argc, argv);
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.view_all( rend.bbox );

    rend.add_manipulator( std::make_shared<arcball_manipulator>(rend.cam, mouse::Left) );
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Middle) );
    // Additional "Alt + LMB" pan manipulator for setups w/o middle mouse button
    rend.add_manipulator( std::make_shared<pan_manipulator>(rend.cam, mouse::Left, keyboard::Alt) );
    rend.add_manipulator( std::make_shared<zoom_manipulator>(rend.cam, mouse::Right) );

    rend.event_loop();
}
