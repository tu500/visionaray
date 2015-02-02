// This file is distributed under the MIT license.
// See the LICENSE file for details.

#include <iomanip>
#include <iostream>
#include <ostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)

#include <AvailabilityMacros.h>

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#else // VSNRAY_OS_DARWIN

#include <GL/gl.h>
#include <GL/glut.h>

#endif

#ifdef FREEGLUT
#include <GL/freeglut_ext.h>
#endif

#include <visionaray/detail/aligned_vector.h>
#include <visionaray/texture/texture.h>
#include <visionaray/utility/kernels.h>
#include <visionaray/utility/params.h>
#include <visionaray/bvh.h>
#ifdef __CUDACC__
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>

#include "manip/arcball_manipulator.h"
#include "manip/pan_manipulator.h"
#include "manip/zoom_manipulator.h"
#include "default_scenes.h"
#include "obj_loader.h"
#include "render_bvh.h"
#include "timer.h"


using std::make_shared;
using std::shared_ptr;

using namespace visionaray;

typedef std::vector<shared_ptr<visionaray::camera_manipulator>> manipulators;

struct renderer
{

//  typedef       float                 scalar_type_cpu;
    typedef simd::float4                scalar_type_cpu;
//  typedef simd::float8                scalar_type_cpu;
    typedef       float                 scalar_type_gpu;
    typedef basic_ray<scalar_type_cpu>  ray_type_cpu;
    typedef basic_ray<scalar_type_gpu>  ray_type_gpu;


    renderer()
        : w(800)
        , h(800)
        , frame(0)
        , rt(visionaray::PF_RGBA32F, visionaray::PF_UNSPECIFIED)
#ifdef __CUDACC__
        , device_rt(visionaray::PF_RGBA32F, visionaray::PF_UNSPECIFIED)
#endif
        , down_button(mouse::NoButton)
    {
    }

    int w;
    int h;
    unsigned frame;

    detail::obj_scene           scene;

    tiled_sched<ray_type_cpu>   sched_cpu;
    cpu_buffer_rt               rt;
#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>    sched_gpu;
    pixel_unpack_buffer_rt      device_rt;
#endif
    shared_ptr<camera>          cam;
    manipulators                manips;

    mouse::button down_button;
    mouse::pos motion_pos;
    mouse::pos down_pos;
    mouse::pos up_pos;

    visionaray::frame_counter counter;
};

renderer* rend = 0;

//auto scene = visionaray::detail::default_generic_prim_scene();


struct configuration
{
    enum device_type
    {
        CPU = 0,
        GPU
    };

    configuration()
        : dev_type(CPU)
        , show_hud(true)
        , show_hud_ext(true)
        , show_bvh(false)
    {
    }

    device_type dev_type;

    bool        show_hud;
    bool        show_hud_ext;
    bool        show_bvh;
};

configuration config;

void render_hud()
{

    auto w = rend->w;
    auto h = rend->h;

    int x = visionaray::clamp( rend->motion_pos.x, 0, w - 1 );
    int y = visionaray::clamp( rend->motion_pos.y, 0, h - 1 );
    vec4* color = static_cast<vec4*>(rend->rt.color());
    vec4 rgba = color[(h - 1 - y) * w + x];

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w * 2, 0, h * 2);

    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    std::stringstream stream;
    stream << "X: " << x;
    std::string str = stream.str();
    glRasterPos2i(10, h * 2 - 34);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "Y: " << y;
    str = stream.str();
    glRasterPos2i(10, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "W: " << w;
    str = stream.str();
    glRasterPos2i(100, h * 2 - 34);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "H: " << h;
    str = stream.str();
    glRasterPos2i(100, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream << std::fixed << std::setprecision(2);

    stream.str(std::string());
    stream << "R: " << rgba.x;
    str = stream.str();
    glRasterPos2i(10, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "G: " << rgba.y;
    str = stream.str();
    glRasterPos2i(100, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "B: " << rgba.z;
    str = stream.str();
    glRasterPos2i(190, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    stream.str(std::string());
    stream << "FPS: " << rend->counter.register_frame();
    str = stream.str();
    glRasterPos2i(10, h * 2 - 136);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

}

void render_hud_ext()
{

    auto w = rend->w;
    auto h = rend->h;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w * 2, 0, h * 2);

    std::stringstream stream;
    stream << "# Triangles: " << rend->scene.primitives.size();
    std::string str = stream.str();
    glRasterPos2i(300, h * 2 - 34);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }

/*    stream.str(std::string());
    stream << "# BVH nodes: " << 1000;
    str = stream.str();
    glRasterPos2i(300, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }*/

    stream.str(std::string());
    stream << "SPP: " << rend->frame;
    str = stream.str();
    glRasterPos2i(300, h * 2 - 68);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }


    stream.str(std::string());
    stream << "Device: " << ( (config.dev_type == configuration::GPU) ? "GPU" : "CPU" );
    str = stream.str();
    glRasterPos2i(300, h * 2 - 102);
    for (size_t i = 0; i < str.length(); ++i)
    {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, str[i]);
    }


    glMatrixMode( GL_MODELVIEW );
    glPushMatrix();
    glLoadIdentity();

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);

}

void display_func()
{

    auto& scene = rend->scene;
    static auto const primitives = scene.primitives;
    static auto const normals    = scene.normals;
    static auto const tex_coords = scene.tex_coords;
    static auto const materials  = scene.materials;
    static auto const textures   = scene.textures;

    typedef point_light<float> light_type;

    visionaray::aligned_vector<light_type> lights
    {
//        light_type{vec3(0.0, 1.0, 1.0)}
        light_type{rend->cam->eye() - rend->cam->center()}
    };

#ifdef __CUDACC__
    thrust::device_vector<light_type> device_lights = lights;
#endif

//    timer t;
    static auto const host_bvh = build(primitives.data(), primitives.size());
//    std::cerr << t.elapsed() << std::endl;

    if (config.dev_type == configuration::GPU)
    {
#ifdef __CUDACC__
        typedef vector<4, float> color_type;

        typedef decltype(scene.primitives)::value_type  primitive_type;
        typedef decltype(scene.normals)::value_type     normal_type;
        typedef decltype(scene.materials)::value_type   material_type;
        typedef device_bvh<primitive_type>              device_bvh_type;

        static device_bvh_type const dbvh(host_bvh);

        device_bvh_type* device_bvh_pointers = 0;
        auto err = cudaMalloc( &device_bvh_pointers, 1 * sizeof(device_bvh_type) );if (err != cudaSuccess) throw std::runtime_error("malloc?");
        err = cudaMemcpy( device_bvh_pointers, &dbvh, 1 * sizeof(device_bvh_type), cudaMemcpyHostToDevice );if (err != cudaSuccess) throw std::runtime_error("memcpy?");

        static thrust::device_vector<normal_type>   const device_normals    = normals;
        static thrust::device_vector<material_type> const device_materials  = materials;

        sched_params<color_type, simple::pixel_sampler_type> sparams
        {
            *rend->cam,
            &rend->device_rt
        };

        auto kparams = simple::make_params
        (
            device_bvh_pointers,
            device_bvh_pointers + 1,
            thrust::raw_pointer_cast(device_normals.data()),
            thrust::raw_pointer_cast(device_materials.data()),
            thrust::raw_pointer_cast(device_lights.data()),
            thrust::raw_pointer_cast(device_lights.data() + device_lights.size())
        );

        typedef vector<4, renderer::scalar_type_gpu> internal_color_type;

        typedef simple::kernel<internal_color_type, decltype(kparams)> kernel_type;
        kernel_type kernel;
        kernel.params = kparams;
        rend->sched_gpu.frame(kernel, sparams /*, ++rend->frame*/ );

        err = cudaFree( device_bvh_pointers );if (err != cudaSuccess) throw std::runtime_error("free?");
#endif
    }
    else if (config.dev_type == configuration::CPU)
    {
#ifndef __CUDA_ARCH__
        typedef vector<4, float> color_type;

        sched_params<color_type, simple::pixel_sampler_type> sparams
        {
            *rend->cam,
            &rend->rt
        };

        auto kparams = simple::make_params
        (
            &host_bvh,
            &host_bvh + 1,
            normals.data(),
//            tex_coords.data(),
            materials.data(),
//            textures.data(),
            lights.data(),
            lights.data() + lights.size()
        );

        typedef vector<4, renderer::scalar_type_cpu> internal_color_type;

        typedef simple::kernel<internal_color_type, decltype(kparams)> kernel_type;
        kernel_type kernel;
        kernel.params = kparams;
        rend->sched_cpu.frame(kernel, sparams /*, ++rend->frame*/ );
#endif
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if (config.dev_type == configuration::GPU && false /* no direct rendering */)
    {
#ifdef __CUDACC__
//        rend->rt = rend->device_rt;
//        rend->rt.display_color_buffer();
#endif
    }
    else if (config.dev_type == configuration::GPU && true /* direct rendering */)
    {
#ifdef __CUDACC__
        rend->device_rt.display_color_buffer();
#endif
    }
    else
    {
        rend->rt.display_color_buffer();
    }

    if (config.show_hud)
    {
        render_hud();
    }

    if (config.show_hud_ext)
    {
        render_hud_ext();
    }

    if (config.show_bvh)
    {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(rend->cam->get_proj_matrix().data());

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadMatrixf(rend->cam->get_view_matrix().data());

        render(host_bvh);

        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    glutSwapBuffers();

}

void idle_func()
{
    glutPostRedisplay();
}

void keyboard_func(unsigned char key, int, int)
{
    switch (key)
    {
    case 'm':
#ifdef __CUDACC__
        if (config.dev_type == configuration::CPU)
        {
            config.dev_type = configuration::GPU;
        }
        else
        {
            config.dev_type = configuration::CPU;
        }
        rend->counter.reset();
        rend->frame = 0;
#endif
        break;

    default:
        break;
    }
}

void mouse_func(int button, int state, int x, int y)
{
    using namespace visionaray::mouse;

    mouse::button b = map_glut_button(button);
    pos p = { x, y };

    if (state == GLUT_DOWN)
    {
        for (auto it = rend->manips.begin(); it != rend->manips.end(); ++it)
        {
            (*it)->handle_mouse_down(visionaray::mouse_event(mouse::ButtonDown, b, p));
        }
        rend->down_pos = p;
        rend->down_button = b;
    }
    else if (state == GLUT_UP)
    {
        for (auto it = rend->manips.begin(); it != rend->manips.end(); ++it)
        {
            (*it)->handle_mouse_up(visionaray::mouse_event(mouse::ButtonUp, b, p));
        }
        rend->up_pos = p;
        rend->down_button = mouse::NoButton;
    }
}

void motion_func(int x, int y)
{
    using namespace visionaray::mouse;

    rend->frame = 0;

    pos p = { x, y };
    for (auto it = rend->manips.begin(); it != rend->manips.end(); ++it)
    {
        (*it)->handle_mouse_move( visionaray::mouse_event(mouse::Move, NoButton, p, rend->down_button, visionaray::keyboard::NoKey) );
    }
    rend->motion_pos = p;
}

void passive_motion_func(int x, int y)
{
    rend->motion_pos = { x, y };
}

void reshape_func(int w, int h)
{
    rend->frame = 0;

    glViewport(0, 0, w, h);
    rend->cam->set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    rend->cam->perspective(45.0f * visionaray::constants::pi<float>() / 180.0f, aspect, 0.001f, 1000.0f);
    rend->rt.resize(w, h);
#ifdef __CUDACC__
    rend->device_rt.resize(w, h);
#endif
    rend->w = w;
    rend->h = h;
}

void close_func()
{
    delete rend;
}

int main(int argc, char** argv)
{

    if (argc == 1)
    {
        std::cerr << "Usage: viewer FILENAME" << std::endl;
        return EXIT_FAILURE;
    }

    rend = new renderer;
    rend->scene = visionaray::load_obj(argv[1]);

    glutInit(&argc, argv);

    glutInitDisplayMode(/*GLUT_3_2_CORE_PROFILE |*/ GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);

    glutInitWindowSize(rend->w, rend->h);
    glutCreateWindow("Visionaray GLUT Viewer");
    glutDisplayFunc(display_func);
    glutIdleFunc(idle_func);
    glutKeyboardFunc(keyboard_func);
    glutMouseFunc(mouse_func);
    glutMotionFunc(motion_func);
    glutPassiveMotionFunc(passive_motion_func);
    glutReshapeFunc(reshape_func);
#ifdef FREEGLUT
    glutCloseFunc(close_func);
#else
    atexit(close_func);
#endif

    if (glewInit() != GLEW_OK)
    {
    }

    rend->cam= make_shared<visionaray::camera>();
    float aspect = rend->w / static_cast<float>(rend->h);
    rend->cam->perspective(45.0f * visionaray::constants::pi<float>() / 180.0f, aspect, 0.001f, 1000.0f);
    rend->cam->view_all( rend->scene.bbox );
    rend->manips.push_back( make_shared<visionaray::arcball_manipulator>(rend->cam, mouse::Left) );
    rend->manips.push_back( make_shared<visionaray::pan_manipulator>(rend->cam, mouse::Middle) );
    rend->manips.push_back( make_shared<visionaray::zoom_manipulator>(rend->cam, mouse::Right) );

    glutMainLoop();

}

