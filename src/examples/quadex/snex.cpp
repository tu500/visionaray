// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <memory>
#include <random>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/camera.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h> // for make_kernel_params(...)
#include <visionaray/material.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#ifdef __CUDACC__
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

#include <visionaray/gl/bvh_outline_renderer.h>

#include <common/call_kernel.h>

// fails with conflicting functions any(__nvbool), visionaray::simd::any(__nvbook)
//#define QUAD_NS snex
#define QUAD_NS visionaray

#define CALCULATE_UV 1

#include "basic_quad.h"
#include "swoop.h"
#include "snex.h"

using namespace visionaray;

using viewer_type = viewer_glut;

// using quad_type = QUAD_NS::basic_quad<float>;
// using quad_type = QUAD_NS::quad_prim<float>;
using quad_type = QUAD_NS::swoop_quad<float>;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    //using host_ray_type = basic_ray<simd::float8>;
    using host_ray_type = basic_ray<float>;

#ifdef __CUDACC__
    using ray_type_gpu              = basic_ray<float>;
    //using device_render_target_type = pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using device_render_target_type = gpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using device_bvh_type           = cuda_index_bvh<quad_type>;
    using device_tex_type           = cuda_texture<vector<4, unorm<8>>, 2>;
    using device_tex_ref_type       = typename device_tex_type::ref_type;
#endif

    renderer()
        : //viewer_type(512, 512, "Visionaray Custom Intersector Example")
         bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , host_sched(8)
#ifdef __CUDACC__
        , device_sched(8, 8)
#endif
    {
        make_sphere();
        init_device_data();
    }

    aabb                                        bbox;
    camera                                      cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    unsigned                                    frame_num   = 0;

    // rendering data

    index_bvh<quad_type>                        bvh;
    aligned_vector<quad_type>                   quads;
    aligned_vector<vec3>                        normals;
    aligned_vector<plastic<float>>              materials;

    gl::bvh_outline_renderer                    outlines;
    bool                                        render_bvh = false;

    bool                                        render_gpu = false;


#ifdef __CUDACC__
    device_bvh_type                             device_bvh;
    thrust::device_vector<vec3>                 device_normals;
    thrust::device_vector<plastic<float>>       device_materials;

    cuda_sched<ray_type_gpu>                    device_sched;
    device_render_target_type                   device_rt;
#endif


    void make_sphere(vector<3, float> center=vector<3, float>(0.f), float radius=1.f)
    {
        int prim_id = 0;
        const int resolution = 20;

        for (int i = 0; i < resolution; ++i) // longitude
        {
            for (int j = 0; j < resolution/4; ++j) // latitude
            {
                // position in the xy plane (on the unit circle)
                vec3 xy_pos1 = vec3(
                        cos(2.f * constants::pi<float>() * i / resolution),
                        sin(2.f * constants::pi<float>() * i / resolution),
                        0);
                vec3 xy_pos2 = vec3(
                        cos(2.f * constants::pi<float>() * (i+1) / resolution),
                        sin(2.f * constants::pi<float>() * (i+1) / resolution),
                        0);

                // the height of the latitudinal coordinate
                vec3 z_pos1 = vec3(0, 0, sin(2.f * constants::pi<float>() * j / resolution));
                vec3 z_pos2 = vec3(0, 0, sin(2.f * constants::pi<float>() * (j+1) / resolution));

                // the radius of the latitudinal circle
                // (the sphere intersected with the xy-plane at height Z_pos)
                float radius1 = cos(2.f * constants::pi<float>() * j / resolution);
                float radius2 = cos(2.f * constants::pi<float>() * (j+1) / resolution);


                // quad on the upper hemisphere
                quad_type s = quad_type::make_quad(
                        (xy_pos1 * radius1 + z_pos1) * radius + center,
                        (xy_pos2 * radius1 + z_pos1) * radius + center,
                        (xy_pos2 * radius2 + z_pos2) * radius + center,
                        (xy_pos1 * radius2 + z_pos2) * radius + center);
                s.prim_id = prim_id++;
                s.geom_id = 0;
                quads.push_back(s);

                normals.push_back(xy_pos1 * radius1 + z_pos1);
                normals.push_back(xy_pos2 * radius1 + z_pos1);
                normals.push_back(xy_pos2 * radius2 + z_pos2);
                normals.push_back(xy_pos1 * radius2 + z_pos2);


                // quad on the lower hemisphere
                quad_type s_neg = quad_type::make_quad(
                        (xy_pos2 * radius1 - z_pos1) * radius + center,
                        (xy_pos1 * radius1 - z_pos1) * radius + center,
                        (xy_pos1 * radius2 - z_pos2) * radius + center,
                        (xy_pos2 * radius2 - z_pos2) * radius + center);
                s_neg.prim_id = prim_id++;
                s_neg.geom_id = 0;
                quads.push_back(s_neg);

                normals.push_back(xy_pos2 * radius1 - z_pos1);
                normals.push_back(xy_pos1 * radius1 - z_pos1);
                normals.push_back(xy_pos1 * radius2 - z_pos2);
                normals.push_back(xy_pos2 * radius2 - z_pos2);
            }
        }

        // quads.clear();
        // normals.clear();
        // quad_type s = quad_type::make_quad(
        //         vec3(2, 2, 2),
        //         vec3(2, 2, 3),
        //         vec3(1, 2, 3),
        //         vec3(1, 2, 2)
        //     );
        // quads.push_back(s);
        // normals.push_back(vec3(0, -1, 0));

        plastic<float> m;
        m.set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
        m.set_cd( from_rgb(1.0f, 0.0f, 0.0f) );
        m.set_cs( from_rgb(1.0f, 1.0f, 1.0f) );
        m.set_ka( 0.0f );
        m.set_kd( 1.0f );
        m.set_ks( 1.0f );
        m.set_specular_exp( 60.f );
        materials.push_back(m);

        bvh = build<index_bvh<quad_type>>(quads.data(), quads.size());
    }


protected:

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);
    void on_close();
    void on_key_press(key_event const& event);

    void init_device_data();

};


//-------------------------------------------------------------------------------------------------
// Display function, contains the rendering kernel
//

void renderer::on_display()
{
    static bool init_outlines = true;

    if (init_outlines)
    {
        outlines.init(bvh);
        init_outlines = false;
    }

    // some setup

    auto sparams = make_sched_params(
//            pixel_sampler::jittered_blend_type{},
            cam,
            host_rt
            );


    // a headlight
    point_light<float> light;
    light.set_cl( vec3(1.0f, 1.0f, 1.0f) );
    light.set_kl( 1.0f );
    light.set_position( cam.eye() );


    aligned_vector<point_light<float>> lights;
    lights.push_back(light);

    aligned_vector<index_bvh<quad_type>::bvh_ref> primitives;
    primitives.push_back(bvh.ref());

    if (render_gpu)
    {
#ifdef __CUDACC__
        thrust::device_vector<renderer::device_bvh_type::bvh_ref> device_primitives;

        device_primitives.push_back(device_bvh.ref());

        thrust::device_vector<point_light<float>> device_lights = lights;

        auto kparams = make_kernel_params(
                normals_per_vertex_binding{},
                thrust::raw_pointer_cast(device_primitives.data()),
                thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
                thrust::raw_pointer_cast(device_normals.data()),
                thrust::raw_pointer_cast(device_materials.data()),
                thrust::raw_pointer_cast(device_lights.data()),
                thrust::raw_pointer_cast(device_lights.data()) + device_lights.size(),
                3,
                1E-3f,
                vec4(background_color(), 1.0f),
                vec4(1.0f)
                );

        call_kernel( Simple, device_sched, kparams, frame_num, cam, device_rt );
#endif
    }
    else
    {
#ifndef __CUDA_ARCH__
        auto kparams = make_kernel_params(
                normals_per_vertex_binding{},
                primitives.data(),
                primitives.data() + primitives.size(),
                normals.data(),
                materials.data(),
                lights.data(),
                lights.data() + lights.size(),
                3,          // num bounces - irrelevant for primary ray shading
                1E-3f,      // a tiny number - also irrelevant for primary ray shading
                vec4(background_color(), 1.0f),
                vec4(1.0f)
                );

        call_kernel( Simple, host_sched, kparams, frame_num, cam, host_rt );
        //simple::kernel<decltype(kparams)> kern;
        //kern.params = kparams;

        //host_sched.frame(kern, sparams, ++frame_num);
#endif
    }


    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (render_gpu)
    {
#ifdef __CUDACC__
        device_rt.display_color_buffer();
#endif
    }
    else
    {
        host_rt.display_color_buffer();
    }

    if (render_bvh)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(cam.get_proj_matrix().data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(cam.get_view_matrix().data());

        outlines.frame();

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }
}

//-------------------------------------------------------------------------------------------------
// keypress event
//

void renderer::on_key_press(key_event const& event)
{
    switch (event.key())
    {

    case 'b':
        render_bvh = !render_bvh;

        break;

#ifdef __CUDACC__
    case 'g':
        render_gpu = !render_gpu;

        break;
#endif

    default:
        break;
    }

    viewer_type::on_key_press(event);
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

#ifdef __CUDACC__
    device_rt.resize(w, h);
#endif

    viewer_type::on_resize(w, h);
}


//-------------------------------------------------------------------------------------------------
// mouse move event
//

void renderer::on_mouse_move(visionaray::mouse_event const& event)
{
    if (event.get_buttons() != mouse::NoButton)
    {
        frame_num = 0;
    }

    viewer_type::on_mouse_move(event);
}

void renderer::on_close()
{
    outlines.destroy();
}


//-------------------------------------------------------------------------------------------------
//
//

void renderer::init_device_data()
{
#ifdef __CUDACC__
    // Copy data to GPU
    try
    {
        device_bvh = device_bvh_type(bvh);
        device_normals = thrust::device_vector<vec3>(normals);
        device_materials = thrust::device_vector<plastic<float>>(materials);
    }
    catch (std::bad_alloc&)
    {
        std::cerr << "GPU memory allocation failed" << std::endl;
        device_bvh = device_bvh_type();
        device_normals.clear();
        device_normals.shrink_to_fit();
        device_materials.clear();
        device_materials.shrink_to_fit();
    }
#endif
}


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

//#include <cuda_gl_interop.h>
int main(int argc, char** argv)
{
    // cudaError_t err = cudaSuccess;
    // int dev = 0;
    // cudaDeviceProp prop;
    // err = cudaChooseDevice(&dev, &prop);
    // if (err != cudaSuccess)
    // {
    //     throw std::runtime_error("choose device");
    // }
    // err = cudaGLSetGLDevice(dev);
    // if (err==cudaErrorSetOnActiveProcess)
    // {
    //     err = cudaDeviceReset();
    //     err = cudaGLSetGLDevice(dev);
    // }
    // if (err != cudaSuccess)
    // {
    //     throw std::runtime_error("set GL device");
    // }
    renderer rend;
    // //rend.init_device_data();
    // std::vector<vec3> t2;
    // t2.push_back(vec3(1.f,2.f,3.f));
    // thrust::device_vector<vec3> t;
    // t = thrust::device_vector<vec3>(t2);
    //return;

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