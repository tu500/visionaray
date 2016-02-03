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

#include <visionaray/detail/platform.h>

#if defined(VSNRAY_OS_DARWIN)

#if MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif

#endif // VSNRAY_OS_DARWIN

#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/bvh.h>
#include <visionaray/simple_buffer_rt.h>
#include <visionaray/kernels.h>
#include <visionaray/point_light.h>
#include <visionaray/scheduler.h>

#if FPSTEST_SCHEDULER == SCHED_TBB
#include <visionaray/experimental/tbb_sched.h>
#endif

#include <common/call_kernel.h>
#include <common/model.h>
#include <common/timer.h>
#include <common/util.h>

#include <fpstest/save_png.h>
#include <fpstest/test_model.h>


using namespace visionaray;

#if FPSTEST_PACKET_SIZE == 1
using scalar_type               = float;
#elif FPSTEST_PACKET_SIZE == 4
using scalar_type               = simd::float4;
#elif FPSTEST_PACKET_SIZE == 8
using scalar_type               = simd::float8;
#endif

using ray_type                  = basic_ray<scalar_type>;

using primitive_type            = model::triangle_list::value_type;
using normal_type               = model::normal_list::value_type;
using material_type             = model::mat_list::value_type;

using render_target_type        = simple_buffer_rt<PF_RGBA32F, PF_UNSPECIFIED>;
using bvh_type                  = index_bvh<primitive_type>;


const int width                 = 800;
const int height                = 800;

#if FPSTEST_ALGO == ALGO_SIMPLE
const algorithm   render_algo          = Simple;
#elif FPSTEST_ALGO == ALGO_WHITTED
const algorithm   render_algo          = Whitted;
#elif FPSTEST_ALGO == ALGO_PATHTRACING
const algorithm   render_algo          = Pathtracing;
#endif


model                           mod;
bvh_type                        host_bvh;

#if FPSTEST_SCHEDULER == SCHED_TILED
tiled_sched<ray_type>           scheduler(get_num_processors());
#elif FPSTEST_SCHEDULER == SCHED_SIMPLE
simple_sched<ray_type>          scheduler;
#elif FPSTEST_SCHEDULER == SCHED_TBB
tbb_sched<ray_type>             scheduler;
#endif

render_target_type              rt;
camera                          cam;
visionaray::frame_counter       fps_counter; // TODO: make frame_counter interval configurable



void render(unsigned int frame)
{
    using light_type = point_light<float>;

    aligned_vector<light_type> lights;

    light_type light;
    light.set_cl( vec3(1.0, 1.0, 1.0) );
    light.set_kl(1.0);
    light.set_position( cam.eye() );

    lights.push_back( light );

    vec3 bgcolor    = { 0.1f, 0.4f, 1.0f };

    auto bounds     = mod.bbox;
    auto diagonal   = bounds.max - bounds.min;
    auto bounces    = render_algo == Pathtracing ? 10U : 4U;
    auto epsilon    = max( 1E-3f, length(diagonal) * 1E-5f );

    aligned_vector<bvh_type::bvh_ref> primitives;

    primitives.push_back(host_bvh.ref());

    auto kparams = make_kernel_params(
            primitives.data(),
            primitives.data() + primitives.size(),
            mod.normals.data(),
            mod.materials.data(),
            lights.data(),
            lights.data() + lights.size(),
            bounces,
            epsilon,
            vec4(bgcolor, 1.0f),
            render_algo == Pathtracing ? vec4(1.0) : vec4(0.0)
            );

    call_kernel( render_algo, scheduler, kparams, frame, cam, rt );
}

void save_to_png_file(std::string filename)
{
    vec4 *colors = rt.color();

    // clip color values
    for (size_t i=0; i<rt.width()*rt.height(); i++)
      for (size_t j=0; j<4; j++)
        if (colors[i][j] > 1.)
          colors[i][j] = 1.;

    save_png(filename, rt.width(), rt.height(), colors);
}

int main(int argc, char** argv)
{
    std::srand(0x13e0d16d); // some arbitrary, random but fixed integer

    // Generate model
    mod = generate_tetrahedrons_cube(1000000);

    // Create the BVH on the host
    host_bvh = build<bvh_type>(mod.primitives.data(), mod.primitives.size());

    rt.resize(width, height);

    float aspect = width / static_cast<float>(height);
    cam.set_viewport(0, 0, width, height);
    cam.perspective(45.0f * constants::degrees_to_radians<float>(), aspect, 0.001f, 1000.0f);
    cam.look_at(vec3(80., 80., 80.), vec3(0.,0.,0.), vec3(0.,1.,0.));

    fps_counter.reset();

    unsigned int frame = 0;

    while (true)
    {
        render(frame);

        double fps;
        if ((fps = fps_counter.register_frame()) != 0.0)
        {
            save_to_png_file("out.png");
            std::cout << "FPS: " << fps << std::endl;
            exit(EXIT_SUCCESS);
        }
    }
}
