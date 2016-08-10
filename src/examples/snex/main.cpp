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

#include <common/manip/arcball_manipulator.h>
#include <common/manip/pan_manipulator.h>
#include <common/manip/zoom_manipulator.h>
#include <common/viewer_glut.h>

#include <visionaray/detail/render_bvh.h>

#include "snex.h"

using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// struct with state variables
//

struct renderer : viewer_type
{
    using host_ray_type = basic_ray<simd::float8>;
    //using host_ray_type = basic_ray<float>;

    renderer()
        : viewer_type(512, 512, "Visionaray Custom Intersector Example")
        , bbox({ -1.0f, -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f })
        , host_sched(8)
    {
        make_sphere();
    }

    aabb                                        bbox;
    camera                                      cam;
    cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>   host_rt;
    tiled_sched<host_ray_type>                  host_sched;

    unsigned                                    frame_num   = 0;

    // rendering data

    index_bvh<snex::quad_prim<float>>           bvh;
    aligned_vector<snex::quad_prim<float>>      quads;
    aligned_vector<vec3>                        normals;
    aligned_vector<plastic<float>>              materials;

    detail::bvh_outline_renderer                outlines;
    bool                                        render_bvh = false;


    void make_sphere(vector<3, float> center=vector<3, float>(0.f), float radius=1.f)
    {
        int prim_id = 0;
        const int resolution = 20;

        for (int i = 0; i < resolution; ++i) // longitude
        {
            for (int j = 0; j < resolution/4 - 1; ++j) // latitude
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

                // the radius of the latitudianl circle
                // (the sphere intersected with the xy-plane at height Z_pos)
                float radius1 = cos(2.f * constants::pi<float>() * j / resolution);
                float radius2 = cos(2.f * constants::pi<float>() * (j+1) / resolution);


                // quad on the upper hemisphere
                snex::quad_prim<float> s(
                        (xy_pos1 * radius1 + z_pos1) * radius + center,
                        (xy_pos2 * radius1 + z_pos1) * radius + center,
                        (xy_pos1 * radius2 + z_pos2) * radius + center,
                        (xy_pos2 * radius2 + z_pos2) * radius + center);
                s.prim_id = prim_id++;
                s.geom_id = 0;
                quads.push_back(s);

                normals.push_back(xy_pos1 * radius1 + z_pos1);
                normals.push_back(xy_pos2 * radius1 + z_pos1);
                normals.push_back(xy_pos1 * radius2 + z_pos2);
                normals.push_back(xy_pos2 * radius2 + z_pos2);


                // quad on the lower hemisphere
                snex::quad_prim<float> s_neg(
                        (xy_pos1 * radius1 - z_pos1) * radius + center,
                        (xy_pos1 * radius2 - z_pos2) * radius + center,
                        (xy_pos2 * radius1 - z_pos1) * radius + center,
                        (xy_pos2 * radius2 - z_pos2) * radius + center);
                s_neg.prim_id = prim_id++;
                s_neg.geom_id = 0;
                quads.push_back(s_neg);

                normals.push_back(xy_pos1 * radius1 - z_pos1);
                normals.push_back(xy_pos1 * radius2 - z_pos2);
                normals.push_back(xy_pos2 * radius1 - z_pos1);
                normals.push_back(xy_pos2 * radius2 - z_pos2);
            }
        }

        plastic<float> m;
        m.set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
        m.set_cd( from_rgb(1.0f, 0.0f, 0.0f) );
        m.set_cs( from_rgb(1.0f, 1.0f, 1.0f) );
        m.set_ka( 0.0f );
        m.set_kd( 1.0f );
        m.set_ks( 1.0f );
        m.set_specular_exp( 60.f );
        materials.push_back(m);

        bvh = build<index_bvh<snex::quad_prim<float>>>(quads.data(), quads.size());
    }


protected:

    void on_display();
    void on_mouse_move(visionaray::mouse_event const& event);
    void on_resize(int w, int h);
    void on_close();
    void on_key_press(key_event const& event);

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

    aligned_vector<index_bvh<snex::quad_prim<float>>::bvh_ref> primitives;
    primitives.push_back(bvh.ref());

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

    simple::kernel<decltype(kparams)> kern;
    kern.params = kparams;

    host_sched.frame(kern, sparams, ++frame_num);


    // display the rendered image

    auto bgcolor = background_color();
    glClearColor(bgcolor.x, bgcolor.y, bgcolor.z, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    host_rt.display_color_buffer();

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
