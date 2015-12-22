// This file is distributed under the MIT license.
// See the LICENSE file for details.

#define SCHED_TILED       0
#define SCHED_SIMPLE      1
#define SCHED_TBB         2

#define ALGO_SIMPLE       0
#define ALGO_WHITTED      1
#define ALGO_PATHTRACING  2

#include <fstream>
#include <iomanip>
#include <iostream>
#include <istream>
#include <ostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include <GL/glew.h>

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#include <OpenGL/gl.h>
#include <GLUT/glut.h>

#else // VSNRAY_OS_DARWIN

#include <GL/gl.h>
#include <GL/glut.h>

#endif

#include <Support/CmdLine.h>
#include <Support/CmdLineUtil.h>

#include <visionaray/gl/util.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/cpu_buffer_rt.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#if FPSTEST_SCHEDULER == SCHED_TBB
#include <visionaray/experimental/tbb_sched.h>
#endif

#ifdef __CUDACC__
#include <visionaray/gpu_buffer_rt.h>
#include <visionaray/pixel_unpack_buffer_rt.h>
#endif

#include <common/call_kernel.h>
#include <common/model.h>
#include <common/obj_loader.h>
#include <common/render_bvh.h>
#include <common/timer.h>
#include <common/util.h>
#include <common/viewer_glut.h>


using namespace visionaray;

using viewer_type = viewer_glut;


//-------------------------------------------------------------------------------------------------
// Renderer, stores state, geometry, normals, ...
//

struct renderer : viewer_type
{

#if FPSTEST_PACKET_SIZE == 1
    using scalar_type_cpu           = float;
#elif FPSTEST_PACKET_SIZE == 4
    using scalar_type_cpu           = simd::float4;
#elif FPSTEST_PACKET_SIZE == 8
    using scalar_type_cpu           = simd::float8;
#endif

    using scalar_type_gpu           = float;
    using ray_type_cpu              = basic_ray<scalar_type_cpu>;
    using ray_type_gpu              = basic_ray<scalar_type_gpu>;

    using primitive_type            = model::triangle_list::value_type;
    using normal_type               = model::normal_list::value_type;
    using material_type             = model::mat_list::value_type;

    using host_render_target_type   = cpu_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using host_bvh_type             = index_bvh<primitive_type>;
#ifdef __CUDACC__
    using device_render_target_type = pixel_unpack_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
    using device_bvh_type           = cuda_index_bvh<primitive_type>;
#endif

    enum device_type
    {
        CPU = 0,
        GPU
    };


    renderer()
        : viewer_type(800, 800, "Visionaray Viewer")

#if FPSTEST_SCHEDULER == SCHED_TILED
        , host_sched(get_num_processors())
#elif FPSTEST_SCHEDULER == SCHED_SIMPLE
        , host_sched()
#elif FPSTEST_SCHEDULER == SCHED_TBB
#error TODO
#endif

    {
        using namespace support;

        add_cmdline_option( cl::makeOption<std::string&>(
            cl::Parser<>(),
            "filename",
            cl::Desc("Input file in wavefront obj format"),
            cl::Positional,
            cl::Required,
            cl::init(this->filename)
            ) );
    }


    int w                       = 800;
    int h                       = 800;
    unsigned    frame           = 0;
    device_type dev_type        = CPU;
    bool        show_hud        = true;
    bool        show_hud_ext    = true;
    bool        show_bvh        = false;

#if FPSTEST_ALGO == ALGO_SIMPLE
    algorithm   algo            = Simple;
#elif FPSTEST_ALGO == ALGO_WHITTED
    algorithm   algo            = Whitted;
#elif FPSTEST_ALGO == ALGO_PATHTRACING
    algorithm   algo            = Pathtracing;
#endif


    std::string filename;
    std::string initial_camera;

    model mod;

    host_bvh_type                           host_bvh;

#ifdef __CUDACC__
    device_bvh_type                         device_bvh;
    thrust::device_vector<normal_type>      device_normals;
    thrust::device_vector<material_type>    device_materials;
#endif

#if FPSTEST_SCHEDULER == SCHED_TILED
    tiled_sched<ray_type_cpu>   host_sched;
#elif FPSTEST_SCHEDULER == SCHED_SIMPLE
    simple_sched<ray_type_cpu>   host_sched;
#elif FPSTEST_SCHEDULER == SCHED_TBB
    tbb_sched<ray_type_cpu>     host_sched;
#endif

    host_render_target_type     host_rt;

#ifdef __CUDACC__
    cuda_sched<ray_type_gpu>    device_sched;
    device_render_target_type   device_rt;
#endif

    camera                      cam;

    visionaray::frame_counter   fps_counter;
    visionaray::timer           test_start_timer;
    bool                        test_start_phase = true;

    bvh_outline_renderer        outlines;

protected:

    void on_close();
    void on_display();
    void on_resize(int w, int h);

private:

};



void renderer::on_close()
{
    outlines.destroy();
}

void renderer::on_display()
{
    using light_type = point_light<float>;

    aligned_vector<light_type> host_lights;

    light_type light;
    light.set_cl( vec3(1.0, 1.0, 1.0) );
    light.set_kl(1.0);
    light.set_position( cam.eye() );

    host_lights.push_back( light );

    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = algo == Pathtracing ? 10U : 4U;
    auto epsilon    = max( 1E-3f, length(diagonal) * 1E-5f );

    vec4 bg_color(0.1, 0.4, 1.0, 1.0);

    if (dev_type == renderer::GPU)
    {
#ifdef __CUDACC__
        thrust::device_vector<renderer::device_bvh_type::bvh_ref> device_primitives;

        device_primitives.push_back(device_bvh.ref());

        thrust::device_vector<light_type> device_lights = host_lights;

        auto kparams = make_kernel_params(
                thrust::raw_pointer_cast(device_primitives.data()),
                thrust::raw_pointer_cast(device_primitives.data()) + device_primitives.size(),
                thrust::raw_pointer_cast(device_normals.data()),
                thrust::raw_pointer_cast(device_materials.data()),
                thrust::raw_pointer_cast(device_lights.data()),
                thrust::raw_pointer_cast(device_lights.data()) + device_lights.size(),
                bounces,
                epsilon,
                vec4(background_color(), 1.0f),
                algo == Pathtracing ? vec4(1.0) : vec4(0.0)
                );

        call_kernel( algo, device_sched, kparams, frame, cam, device_rt );
#endif
    }
    else if (dev_type == renderer::CPU)
    {
#ifndef __CUDA_ARCH__
        aligned_vector<renderer::host_bvh_type::bvh_ref> host_primitives;

        host_primitives.push_back(host_bvh.ref());

        auto kparams = make_kernel_params(
                host_primitives.data(),
                host_primitives.data() + host_primitives.size(),
                mod.normals.data(),
//              mod.tex_coords.data(),
                mod.materials.data(),
//              mod.textures.data(),
                host_lights.data(),
                host_lights.data() + host_lights.size(),
                bounces,
                epsilon,
                vec4(background_color(), 1.0f),
                algo == Pathtracing ? vec4(1.0) : vec4(0.0)
                );

        call_kernel( algo, host_sched, kparams, frame, cam, host_rt );
#endif
    }

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glEnable(GL_FRAMEBUFFER_SRGB);

    if (dev_type == renderer::GPU && false /* no direct rendering */)
    {
#ifdef __CUDACC__
//        host_rt = device_rt;
//        host_rt.display_color_buffer();
#endif
    }
    else if (dev_type == renderer::GPU && true /* direct rendering */)
    {
#ifdef __CUDACC__
        device_rt.display_color_buffer();
#endif
    }
    else
    {
        host_rt.display_color_buffer();
    }


    // OpenGL overlay rendering

    glColor3f(1.0f, 1.0f, 1.0f);


    // FPS measurement
    if (test_start_phase)
    {
        // wait some time
        if (test_start_timer.elapsed() > 1.0)
        {
            test_start_phase = false;
            fps_counter.reset();
        }
    }

    else
    {
        double fps;

        // check if there is enough data
        // TODO: maybe want to configure measured timespan, this needs changes in the counter class
        if ((fps = fps_counter.register_frame()) != 0.0)
        {
            std::cout << "FPS: " << fps << std::endl;
            // FIXME this causes an error, for now use exit...
            //quit();
            exit(EXIT_SUCCESS);
        }
    }

}

void renderer::on_resize(int w, int h)
{
    frame = 0;

    cam.set_viewport(0, 0, w, h);
    float aspect = w / static_cast<float>(h);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    host_rt.resize(w, h);
#ifdef __CUDACC__
    device_rt.resize(w, h);
#endif
    viewer_type::on_resize(w, h);
}

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

    gl::init_debug_callback();

    // Load the scene
    //std::cout << "Loading model...\n";

    try
    {
        visionaray::load_obj(rend.filename, rend.mod);
    }
    catch (std::exception& e)
    {
        std::cerr << "Failed loading obj model: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

//  timer t;

    //std::cout << "Creating BVH...\n";

    // Create the BVH on the host
    rend.host_bvh = build<renderer::host_bvh_type>(rend.mod.primitives.data(), rend.mod.primitives.size());

    //std::cout << "Ready\n";

#ifdef __CUDACC__
    // Copy data to GPU
    try
    {
        rend.device_bvh = renderer::device_bvh_type(rend.host_bvh);
        rend.device_normals = rend.mod.normals;
        rend.device_materials = rend.mod.materials;
    }
    catch (std::bad_alloc&)
    {
        std::cerr << "GPU memory allocation failed" << std::endl;
        rend.device_bvh = renderer::device_bvh_type();
        rend.device_normals.resize(0);
        rend.device_materials.resize(0);
    }
#endif

//  std::cout << t.elapsed() << std::endl;

    float aspect = rend.width() / static_cast<float>(rend.height());

    rend.cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    rend.cam.view_all( rend.mod.bbox );

    rend.toggle_full_screen();

    rend.test_start_timer.reset();

    rend.event_loop();

}
