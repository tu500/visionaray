// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <cstring> // memset
#include <memory>

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


//-------------------------------------------------------------------------------------------------
// Main function, performs initialization
//

int main(int argc, char** argv)
{
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
