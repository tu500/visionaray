
#include <visionaray/aligned_vector.h>

#include <common/model.h>

using namespace visionaray;

float random_float(float lower_bound, float upper_bound)
{
  return lower_bound + static_cast<float>(rand()) / static_cast<float>(RAND_MAX/(upper_bound-lower_bound));
}

model generate_tetrahedrons_cube(size_t tetrahedron_count)
{
    aligned_vector<basic_triangle<3, float>>       triangles;
    aligned_vector<vec3>                           normals;
    aligned_vector<plastic<float>>                 materials;

    triangles.reserve(tetrahedron_count * 4 + 12);
    normals.reserve(tetrahedron_count * 4 + 12);

    // set default materials
    plastic<float> mat;
    mat.set_ca( from_rgb(0.0f, 0.0f, 0.0f) );
    mat.set_cd( from_rgb(0.8f, 0.2f, 0.1f) );
    mat.set_cs( from_rgb(1.0f, 0.9f, 0.7f) );
    mat.set_ka( 1.0f );
    mat.set_kd( 1.0f );
    mat.set_ks( 1.0f );
    mat.set_specular_exp( 10.0f );
    materials.push_back(mat);

    plastic<float> mat2;
    mat2.set_ca( from_rgb(0.0f, 0.0f, 0.03f) );
    mat2.set_cd( from_rgb(0.0f, 0.0f, 0.99f) );
    mat2.set_cs( from_rgb(0.7f, 0.9f, 1.19f) );
    mat2.set_ka( 1.0f );
    mat2.set_kd( 1.0f );
    mat2.set_ks( 1.0f );
    mat2.set_specular_exp( 162.7f );
    materials.push_back(mat2);

    // Generate tetrahedrons

    std::array<vec3, 4> std_tetrahedron = {{
        { 1.,  1.,  1.},
        {-1., -1.,  1.},
        {-1.,  1., -1.},
        { 1., -1., -1.},
    }};

    for (size_t i = 0; i<tetrahedron_count; i++)
    {
        // random rotation
        float angle = random_float(0, 2 * M_PI);
        vec3 axis = {
            random_float(-1,1),
            random_float(-1,1),
            random_float(0.00001,1)
        };

        // get rotation matrix
        mat4 rmat4 = rotation(rotation(axis, angle));
        matrix<3, 3, float> rmat;
        rmat(0,0) = rmat4(0,0);
        rmat(0,1) = rmat4(0,1);
        rmat(0,2) = rmat4(0,2);
        rmat(1,0) = rmat4(1,0);
        rmat(1,1) = rmat4(1,1);
        rmat(1,2) = rmat4(1,2);
        rmat(2,0) = rmat4(2,0);
        rmat(2,1) = rmat4(2,1);
        rmat(2,2) = rmat4(2,2);

        // random scale, position
        float scale = random_float(6,20);
        vec3 translate = {
            random_float(-100, 100),
            random_float(-100, 100),
            random_float(-100, 100),
        };

        // final tetrahedron vertices
        std::array<vec3, 4> final_th = {{
            rmat * std_tetrahedron[0] / scale + translate,
            rmat * std_tetrahedron[1] / scale + translate,
            rmat * std_tetrahedron[2] / scale + translate,
            rmat * std_tetrahedron[3] / scale + translate,
        }};

        // store triangles
        triangles.emplace_back(
                final_th[0],
                final_th[3] - final_th[0],
                final_th[2] - final_th[0]
                );
        triangles.emplace_back(
                final_th[2],
                final_th[3] - final_th[2],
                final_th[1] - final_th[2]
                );
        triangles.emplace_back(
                final_th[1],
                final_th[0] - final_th[1],
                final_th[2] - final_th[1]
                );
        triangles.emplace_back(
                final_th[0],
                final_th[1] - final_th[0],
                final_th[3] - final_th[0]
                );

    }


    // set prim_id to identify the triangle
    unsigned prim_id = 0;

    // set geometry id to map to triangles to materials
    for (auto& tri : triangles)
    {
        tri.prim_id = prim_id++;

        // all have the same material and texture
        tri.geom_id = 0;
    }


    // Generate Cube
    // vertices
    std::array<vec3, 8> cv = {{
        { 105.,  105.,  105.},
        { 105., -105.,  105.},
        {-105., -105.,  105.},
        {-105.,  105.,  105.},
        { 105.,  105., -105.},
        { 105., -105., -105.},
        {-105., -105., -105.},
        {-105.,  105., -105.},
    }};

    // triangles
    triangles.emplace_back(cv[0], cv[1] - cv[0], cv[2] - cv[0]);
    triangles.emplace_back(cv[0], cv[2] - cv[0], cv[3] - cv[0]);
    triangles.emplace_back(cv[0], cv[4] - cv[0], cv[5] - cv[0]);
    triangles.emplace_back(cv[0], cv[5] - cv[0], cv[1] - cv[0]);
    triangles.emplace_back(cv[1], cv[5] - cv[1], cv[6] - cv[1]);
    triangles.emplace_back(cv[1], cv[6] - cv[1], cv[2] - cv[1]);
    triangles.emplace_back(cv[2], cv[6] - cv[2], cv[7] - cv[2]);
    triangles.emplace_back(cv[2], cv[7] - cv[2], cv[3] - cv[2]);
    triangles.emplace_back(cv[3], cv[7] - cv[3], cv[4] - cv[3]);
    triangles.emplace_back(cv[3], cv[4] - cv[3], cv[0] - cv[3]);
    triangles.emplace_back(cv[5], cv[4] - cv[5], cv[7] - cv[5]);
    triangles.emplace_back(cv[5], cv[7] - cv[5], cv[6] - cv[5]);

    for (size_t i=0; i<12; i++)
    {
        triangles[tetrahedron_count + i].prim_id = prim_id++;
        triangles[tetrahedron_count + i].geom_id = 1;
    }


    // calculate normals
    for (auto const& tri : triangles)
    {
        normals.emplace_back( normalize(cross(tri.e1, tri.e2)) );
    }


    // contstruct model object
    model m;
    m.materials = materials;
    m.primitives = triangles;
    m.normals = normals;

    m.bbox = aabb(vec3(-105.,-105.,-105), vec3(105.,105.,105.));

    return m;
}

