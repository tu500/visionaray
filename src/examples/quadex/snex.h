#pragma once

#include <visionaray/math/math.h>
#include <visionaray/tags.h>
#include <visionaray/bvh.h>

#include <iostream>
#include <iomanip>

namespace visionaray
{

template <typename T>
class quad_prim : public visionaray::primitive<unsigned>
{
public:

    using scalar_type =  T;
    using vec_type    =  vector<3, T>;

public:

    // vertex ordering:
    //
    //    v3
    //       x
    //       | \
    //       |   \
    //       |     \
    //       |       \
    //       |         \
    //       |           x  v4
    //       |           |
    //       |           |
    //       |           |
    //       |           |
    //       x-----------x
    //    v1                v2
    //
    quad_prim() = default;
    quad_prim(
            vector<3, T> const& v1,
            vector<3, T> const& v2,
            vector<3, T> const& v3,
            vector<3, T> const& v4
            )
        : v1(v1)
        , e1(v2-v1)
        , e2(v3-v1)
    {
        // if this quad should be degenerated to a triangle, make sure that v3 == v4
        // v2 == v4 would also be possible, but would need additional checks
        // and a rewrite of the following code

        // edge case, degenerated quad
        if (v3 == v4)
        {
            // ensure this quad is exactly a triangle
            this->v4= vector<2, T>(0., 1.);
            return;
        }

        // calculate (u,v)-coordinates of the 4th vector relative to the triangle (v1, e1, e2)

        vec_type d = v4-v1;
        vec_type s2 = cross(d, e1); // this is a normal of the triangle, use as ray dir

        vec_type s1 = cross(s2, e2);
        scalar_type div = dot(s1, e1);
        scalar_type inv_div = T(1.0) / div;

        scalar_type b1 = dot(d, s1) * inv_div;
        scalar_type b2 = dot(s2, s2) * inv_div;

        this->v4 = vector<2, T>(b1, b2);
    }

    vec_type v1;
    vec_type e1;
    vec_type e2;
    vector<2, scalar_type> v4;

    // need to reorder vertices for degenerated quads
    //
    // vertex ordering:
    //
    //    v4
    //       x
    //       | \
    //       |   \
    //       |     \
    //       |       \
    //       |         \
    //       |           x  v3
    //       |           |
    //       |           |
    //       |           |
    //       |           |
    //       x-----------x
    //    v1                v2
    //
    static quad_prim<float> make_quad(
                vector<3, float> const& v1,
                vector<3, float> const& v2,
                vector<3, float> const& v3,
                vector<3, float> const& v4,
                float epsilon=0.01f
            )
    {
        // ensure degenerated quads are exactly a triangles and the matching
        // vertices are v3 and v4
        if (norm(v1-v2) < epsilon)
            return quad_prim<float>(v3, v4, v2, v2);
        else if (norm(v2-v3) < epsilon)
            return quad_prim<float>(v4, v1, v3, v3);
        else if (norm(v3-v4) < epsilon)
            return quad_prim<float>(v1, v2, v4, v4);
        else if (norm(v4-v1) < epsilon)
            return quad_prim<float>(v2, v3, v1, v1);

        else
            return quad_prim<float>(v1, v2, v4, v3);
    }
};


template <typename T, typename U>
VSNRAY_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect_opt(
        basic_ray<T> const&                     ray,
        quad_prim<U> const&               quad
        )
{

    typedef vector<3, T> vec_type;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.t = T(-1.0);

    // case T != U
    vec_type v1(quad.v1);
    vec_type e1(quad.e1);
    vec_type e2(quad.e2);
    vector<2, T> v4(quad.v4);

    vec_type s1 = cross(ray.dir, e2);
    T div = dot(s1, e1);

    result.hit = ( div != T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    T inv_div = T(1.0) / div;

    vec_type d = ray.ori - v1;
    T b1 = dot(d, s1) * inv_div;

    result.hit &= ( b1 >= T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    vec_type s2 = cross(d, e1);
    T b2 = dot(ray.dir, s2) * inv_div;

    result.hit &= ( b2 >= T(0.0) );

    if ( !any(result.hit) )
    {
        return result;
    }

    // (b1,b2) are (u,v)-coordinates (relative to the triangle (v1,e1,e2)) of
    // the intersection point - check if they lie inside quad

    result.hit &= ( b2 <= ((v4.y-1.0) / v4.x) * b1 + 1.0 ) || v4.x == 0.0;
    result.hit &= ( b1 <= ((v4.x-1.0) / v4.y) * b2 + 1.0 );

    if ( !any(result.hit) )
    {
        return result;
    }

    // now calculate bilinear coordinates relative to the quad
    T u, v;

    // special cases:
    if (quad.v4.x == 1.0)
    {
        u = b1;
        v = b2 / (u * (v4.y - 1.0) + 1.0);
    }
    else if (quad.v4.y == 1.0)
    {
        v = b2;
        u = b1 / (v * (v4.x - 1.0) + 1.0);
    }
    else
    {
        // solve A*u^2 + B*u + C = 0
        T A = -(v4.y - 1.0);
        T B = b1 * (v4.y - 1.0) - b2 * (v4.x - 1.0) - 1.0;
        T C = b1;

        T D = B * B - 4.0 * A * C;
        T Q = -0.5 * (B + copysign(sqrt(D), B));

        u = Q / A;
        u = select(u < 0.0 || u > 1.0, C / Q, u);
        v = b2 / (u * (v4.y - 1.0) + 1.0);
    }

    result.prim_id = quad.prim_id;
    result.geom_id = quad.geom_id;
    result.t = dot(e2, s2) * inv_div;
    result.u = u;
    result.v = v;
    return result;

}

struct quad_intersector_opt : basic_intersector<quad_intersector_opt>
{
    using basic_intersector<quad_intersector_opt>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            quad_prim<S> const& quad
            )
        -> decltype( detail::intersect_opt(ray, quad) )
    {
        return detail::intersect_opt(ray, quad);
    }
};

}
