#pragma once

#include <cstddef>
#include <type_traits>
#include <iostream>

#include <visionaray/math/math.h>
#include <visionaray/intersector.h>

#define CALCULATE_UV 0

namespace visionaray
{

//-------------------------------------------------------------------------------------------------
// Quad primitive
//

template <typename T>
struct basic_quad : primitive<unsigned>
{
    vector<3, T> v1;
    vector<3, T> v2;
    vector<3, T> v3;
    vector<3, T> v4;
};


namespace detail
{

//-------------------------------------------------------------------------------------------------
// Misc. helpers
//

// remove max element from vec3 ---------------------------

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
vector<2, T> remove_max_element(vector<3, T> const& v)
{
    auto tmp = unpack(v);

    // Move max element of each vector to back
    for (size_t i = 0; i < tmp.size(); ++i)
    {
        auto& vv = tmp[i];
        auto max_idx = max_index(vv);

        if (max_idx == 0)
        {
            std::swap(vv.x, vv.z);
        }
        else if (max_idx == 1)
        {
            std::swap(vv.y, vv.z);
        }
    }

    return simd::pack(tmp).xy();
}

// remove element at index from vec3 ----------------------

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type,
    typename I // simd::int_type!
    >
VSNRAY_CPU_FUNC
vector<2, T> remove_at_index(vector<3, T> const& v, I const& index)
{
    using int_array = typename simd::aligned_array<I>::type;

    auto tmp = unpack(v);
    int_array idx;
    store(idx, index);

    for (size_t i = 0; i < tmp.size(); ++i)
    {
        auto& vv = tmp[i];

        if (idx[i] == 0)
        {
            std::swap(vv.x, vv.z);
        }
        else if (idx[i] == 1)
        {
            std::swap(vv.y, vv.z);
        }
    }

    return simd::pack(tmp).xy();
}

VSNRAY_FUNC
vec2 remove_at_index(vec3 const& v, int const& index)
{
    if (index == 0)
        return vec2(v.y, v.z);

    if (index == 1)
        return vec2(v.x, v.z);

    if (index == 2)
        return vec2(v.x, v.y);
}


// SIMD max_index for vec3 --------------------------------

template <
    typename T,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
inline auto max_index(vector<3, T> const& u)
    -> typename simd::int_type<T>::type
{
    using I = typename simd::int_type<T>::type;
    using int_array = typename simd::aligned_array<I>::type;

    int_array i;

    auto tmp = unpack(u);

    for (size_t d = 0; d < tmp.size(); ++d)
    {
        i[d] = select(tmp[d].y < tmp[d].x, 0, 1);
        i[d] = select(tmp[d][i[d]] < tmp[d].z, 2, i[d]);
    }

    return I(i);
}


template <typename T>
struct line
{
    VSNRAY_FUNC line() = default;
    VSNRAY_FUNC line(vector<2, T> const& vv1, vector<2, T> const& vv2)
        : v1(vv1)
        , v2(vv2)
    {
    }

    vector<2, T> v1;
    vector<2, T> v2;
};

// intersect two line segments (1st is parallel to y-axis)!
template <typename T>
VSNRAY_CPU_FUNC
inline auto intersect(vector<2, T> const& p, line<T> const& l)
    -> typename simd::int_type<T>::type
{
    using I = typename simd::int_type<T>::type;

    auto a = l.v1;
    auto b = l.v2;

    return select(
            ((a.x < p.x && p.x < b.x) && ((b.y * (a.x - p.x) + a.y * (p.x - b.x)) >= (a.x - b.x) * p.y))
         || ((b.x < p.x && p.x < a.x) && ((b.y * (a.x - p.x) + a.y * (p.x - b.x)) <= (a.x - b.x) * p.y)),
            I(1),
            I(0)
            );
}


//-------------------------------------------------------------------------------------------------
// Get uv
//

template <typename T>
VSNRAY_FUNC
vector<2, T> get_uv(basic_quad<float> const& quad, vector<3, T> const& isect_pos)
{
    // Glassner 1989, "An Introduction to Ray Tracing", p.60

    using V = vector<3, T>;

    vector<2, T> uv;

    // possibly precalculate ------------------------------

    V e1(quad.v2 - quad.v1);
    V e2(quad.v3 - quad.v2);

    V P_n = cross(e1, e2);
    V P_a(quad.v1 - quad.v2 + quad.v3 - quad.v4);
    V P_b(quad.v2 - quad.v1);
    V P_c(quad.v4 - quad.v1);
    V P_d(quad.v1);

    V N_a = cross(P_a, P_n);
    V N_b = cross(P_b, P_n);
    V N_c = cross(P_c, P_n);

    T D_u0 = dot(N_c, P_d);
    T D_u1 = dot(N_a, P_d) + dot(N_c, P_b);
    T D_u2 = dot(N_a, P_b);

    T D_v0 = dot(N_b, P_d);
    T D_v1 = dot(N_a, P_d) + dot(N_b, P_c);
    T D_v2 = dot(N_a, P_c);

    //-----------------------------------------------------


    // Compute the distance to the plane perpendicular to
    // the quad's "u-axis"...
    //
    // D(u) = (N_c + N_a * u) . (P_d + P_b * u)
    //
    // ... with regards to isect_pos:

    V R_i = isect_pos;

    //
    // D_r(u) = (N_c + N_a * u) . R_i
    //
    // by letting D(u) = D_r(u) and solving the corresponding
    // quadratic equation.

    V Q_ux = N_a / (T(2.0) * D_u2);
    T D_ux = -D_u1 / (T(2.0) * D_u2);
    V Q_uy = -N_c / D_u2;
    T D_uy = D_u0 / D_u2;


    T K_a = D_ux + dot(Q_ux, R_i);
    T K_b = D_uy + dot(Q_uy, R_i);


    auto parallel_u = (D_u2 == T(0.0));
    uv.x = select(
            parallel_u,
            (dot(N_c, R_i) - D_u0) / (D_u1 - dot(N_a, R_i)),
            K_a - sqrt(K_a * K_a - K_b)
            );

    uv.x = select(
            !parallel_u && (uv.x < T(0.0) || uv.x > T(1.0)),
            K_a + sqrt(K_a * K_a - K_b),
            uv.x
            );


    // Do the same for v

    V Q_vx = N_a / (T(2.0) * D_v2);
    T D_vx = -D_v1 / (T(2.0) * D_v2);
    V Q_vy = -N_b / D_v2;
    T D_vy = D_v0 / D_v2;


    K_a = D_vx + dot(Q_vx, R_i);
    K_b = D_vy + dot(Q_vy, R_i);


    auto parallel_v = (D_v2 == T(0.0));
    uv.y = select(
            parallel_v,
            (dot(N_b, R_i) - D_v0) / (D_v1 - dot(N_a, R_i)),
            K_a - sqrt(K_a * K_a - K_b)
            );

    uv.y = select(
            !parallel_v && (uv.y < T(0.0) || uv.y > T(1.0)),
            K_a + sqrt(K_a * K_a - K_b),
            uv.y
            );


    return uv;
}


//-------------------------------------------------------------------------------------------------
// intersect by using test with Pluecker coordinates (cf. Shevtsov et al. 2007,
//      "Ray-Triangle Intersection Algorithm for Modern CPU Architectures"
// TODO: implement/test the precalculations proposed in the paper
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect_pluecker(R const& ray, basic_quad<float> const& quad)
{
    using T  = typename R::scalar_type;
    using V  = vector<3, T>;
    using V2 = vector<2, T>;

    hit_record<R, primitive<unsigned>> result;

    vector<6, T> e1(quad.v1 - quad.v2, cross(quad.v1, quad.v2));
    vector<6, T> e2(quad.v2 - quad.v3, cross(quad.v2, quad.v3));
    vector<6, T> e3(quad.v3 - quad.v4, cross(quad.v3, quad.v4));
    vector<6, T> e4(quad.v4 - quad.v1, cross(quad.v4, quad.v1));
    vector<6, T> r(cross(ray.dir, ray.ori), ray.dir);

    //vector<3, T> e1(cross(vector<3,T>(quad.v1 - quad.v2), vector<3,T>(quad.v2) - ray.ori));
    //vector<3, T> e2(cross(vector<3,T>(quad.v2 - quad.v3), vector<3,T>(quad.v3) - ray.ori));
    //vector<3, T> e3(cross(vector<3,T>(quad.v3 - quad.v4), vector<3,T>(quad.v4) - ray.ori));
    //vector<3, T> e4(cross(vector<3,T>(quad.v4 - quad.v1), vector<3,T>(quad.v1) - ray.ori));
    //vector<3, T> r(ray.dir);

    T s1 = copysign(T(1.0), dot(e1, r));
    T s2 = copysign(T(1.0), dot(e2, r));
    T s3 = copysign(T(1.0), dot(e3, r));
    T s4 = copysign(T(1.0), dot(e4, r));

    result.hit = s1 == s2 && s1 == s3 && s1 == s4;

    //T s1 = dot(e1, r);
    //T s2 = dot(e2, r);
    //T s3 = dot(e3, r);
    //T s4 = dot(e4, r);

    //result.hit = (0 == 0x7fffffff & (
    //    (reinterpret_cast<uint32_t>(s1) ^ reinterpret_cast<uint32_t>(s2)) |
    //    (reinterpret_cast<uint32_t>(s2) ^ reinterpret_cast<uint32_t>(s3)) |
    //    (reinterpret_cast<uint32_t>(s3) ^ reinterpret_cast<uint32_t>(s4))) );

    if (any(result.hit))
    {
        V v1(quad.v1);
        V e1(quad.v2 - quad.v1);
        V e2(quad.v3 - quad.v2);

        V n = normalize(cross(e1, e2));

        T div = dot(n, ray.dir);

        // ray/plane intersection
        result.t = select(
                div != T(0.0),
                dot(v1 - ray.ori, n) / div,
                result.t
                );
        result.prim_id = quad.prim_id;
        result.geom_id = quad.geom_id;
        result.isect_pos = ray.ori + ray.dir * result.t;
#if CALCULATE_UV
        V2 uv = get_uv(quad, result.isect_pos);
        result.u = uv.x;
        result.v = uv.y;
#endif
    }

    return result;
}


//-------------------------------------------------------------------------------------------------
// intersect by projecting quad edges to 2D (cf. Glassner 1989,
//      "An Introduction to Ray Tracing", p. 55
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect_project_2D(R const& ray, basic_quad<float> const& quad)
{
    using T  = typename R::scalar_type;
    using I  = typename simd::int_type<T>::type;
    using V  = vector<3, T>;
    using V2 = vector<2, T>;
    using L  = line<T>;

    hit_record<R, primitive<unsigned>> result;

    V v1(quad.v1);
    V v2(quad.v2);
    V v3(quad.v3);
    V v4(quad.v4);

    V e1(v2 - v1);
    V e2(v3 - v2);

    V n = normalize(cross(e1, e2));

    T div = dot(n, ray.dir);

    // ray/plane intersection
    T t = select(
            div != T(0.0),
            dot(v1 - ray.ori, n) / div,
            t
            );

    result.hit = div != T(0.0) && t >= T(0.0);
    result.t = t;

    if (!any(result.hit))
    {
        return result;
    }

    V isect_pos = ray.ori + ray.dir * t;


    // Project to 2D by throwing away max component of plane eq.
    I index = max_index(n);

    V2 ip_2 = remove_at_index(isect_pos, index);

    V2 v1_2 = remove_at_index(v1, index);
    V2 v2_2 = remove_at_index(v2, index);
    V2 v3_2 = remove_at_index(v3, index);
    V2 v4_2 = remove_at_index(v4, index);

    I num_intersections(0);
    num_intersections += intersect(ip_2, L(v1_2, v2_2));
    num_intersections += intersect(ip_2, L(v2_2, v3_2));
    num_intersections += intersect(ip_2, L(v3_2, v4_2));
    num_intersections += intersect(ip_2, L(v4_2, v1_2));

    result.hit &= num_intersections == 1;
    result.prim_id = quad.prim_id;
    result.geom_id = quad.geom_id;
    result.isect_pos = isect_pos;

#if CALCULATE_UV
    if (!any(result.hit))
    {
        return result;
    }

    V2 uv = get_uv(quad, isect_pos);
    result.u = uv.x;
    result.v = uv.y;
#endif

    return result;
}


//-------------------------------------------------------------------------------------------------
// intersect by unconditionally calculating uv and checking if they are in [0..1]
//      "Schlick and Subrenat 1995 (?)"
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect_uv(R const& ray, basic_quad<float> const& quad)
{
    using T  = typename R::scalar_type;
    using V  = vector<3, T>;
    using V2 = vector<2, T>;

    hit_record<R, primitive<unsigned>> result;

    V v1(quad.v1);
    V e1(quad.v2 - quad.v1);
    V e2(quad.v3 - quad.v2);

    V n = normalize(cross(e1, e2));

    T div = dot(n, ray.dir);

    // ray/plane intersection
    T t = select(
            div != T(0.0),
            dot(v1 - ray.ori, n) / div,
            T(-1.0)
            );

    result.hit = div != T(0.0) && t >= T(0.0);
    result.t = t;

    if (!any(result.hit))
    {
        return result;
    }

    V isect_pos = ray.ori + ray.dir * t;


    V2 uv = get_uv(quad, isect_pos);

    result.hit = uv.x >= T(0.0) && uv.x <= T(1.0) && uv.y >= T(0.0) && uv.y <= T(1.0);
    result.prim_id = quad.prim_id;
    result.geom_id = quad.geom_id;
    result.isect_pos = isect_pos;
    result.u = uv.x;
    result.v = uv.y;

    return result;
}


template <typename T, typename U>
VSNRAY_FUNC
inline hit_record<basic_ray<T>, primitive<unsigned>> intersect_mt_bl_uv(
        basic_ray<T> const&                     ray,
        basic_quad<U> const&                    quad
        )
{

    typedef vector<3, T> vec_type;

    hit_record<basic_ray<T>, primitive<unsigned>> result;
    result.t = T(-1.0);

    // case T != U
    vec_type v1(quad.v1);
    vec_type e1(quad.v2 - quad.v1);
    vec_type e2(quad.v4 - quad.v1);

    // actual intersection test
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

    result.hit &= ( b1 >= T(0.0));

    if ( !any(result.hit) )
    {
        return result;
    }

    vec_type s2 = cross(d, e1);
    T b2 = dot(ray.dir, s2) * inv_div;

    result.hit &= ( b2 >= T(0.0));

    if ( !any(result.hit) )
    {
        return result;
    }

    // calculate (u,v)-coordinates of the 4th quad-vector relative to the triangle (v1, e1, e2)

    vector<3, U> quad_e1 = quad.v2 - quad.v1;
    vector<3, U> quad_e2 = quad.v4 - quad.v1;

    vector<3, U> quad_d = quad.v3 - quad.v1;
    vector<3, U> quad_s2 = cross(quad_d, quad_e1); // this is a normal of the triangle, use as ray dir

    vector<3, U> quad_s1 = cross(quad_s2, quad_e2);
    U quad_div = dot(quad_s1, quad_e1);
    U quad_inv_div = U(1.0) / quad_div;

    U quad_v4_x = dot(quad_d, quad_s1) * quad_inv_div;
    U quad_v4_y = dot(quad_s2, quad_s2) * quad_inv_div;

    // convert to simd type
    T v4_x = T(dot(quad_d, quad_s1) * quad_inv_div);
    T v4_y = T(dot(quad_s2, quad_s2) * quad_inv_div);

    // (b1,b2) are (u,v)-coordinates (relative to the triangle (v1,e1,e2)) of
    // the intersection point - check if they lie inside quad

    result.hit &= ( b2 <= ((v4_y-1.0) / v4_x) * b1 + 1.0 );
    result.hit &= ( b1 <= ((v4_x-1.0) / v4_y) * b2 + 1.0 );

    if ( !any(result.hit) )
    {
        return result;
    }

#if CALCULATE_UV
    // now calculate bilinear coordinates relative to the quad
    T u, v;

    // special cases:
    if (quad_v4_x == 1)
    {
        u = b1;
        v = b2 / (u * (v4_y - 1.0) + 1.0);
    }
    else if (quad_v4_y == 1)
    {
        v = b2;
        u = b1 / (v * (v4_x - 1.0) + 1.0);
    }
    else
    {
        // solve A*u^2 + B*u + C = 0
        T A = -(v4_y - 1.0);
        T B = b1 * (v4_y - 1.0) - b2 * (v4_x - 1.0) - 1.0;
        T C = b1;

        T D = B * B - 4.0 * A * C;
        T Q = -0.5 * (B + copysignf(sqrtf(D), B));

        u = Q / A;
        u = select(u < 0.0 || u > 1.0, C / Q, u);
        v = b2 / (u * (v4_y - 1.0) + 1.0);
    }

    result.u = u;
    result.v = v;
#endif

    result.prim_id = quad.prim_id;
    result.geom_id = quad.geom_id;
    result.t = dot(e2, s2) * inv_div;
    return result;

}

} // detail


//-------------------------------------------------------------------------------------------------
// Interface
//

template <typename R>
VSNRAY_FUNC
hit_record<R, primitive<unsigned>> intersect(R const& ray, basic_quad<float> const& quad)
{
    return detail::intersect_mt_bl_uv(ray, quad);
    //return detail::intersect_pluecker(ray, quad);
    //return detail::intersect_project_2D(ray, quad);
    //return detail::intersect_uv(ray, quad);
}

struct quad_intersector_mt_bl_uv : basic_intersector<quad_intersector_mt_bl_uv>
{
    using basic_intersector<quad_intersector_mt_bl_uv>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_mt_bl_uv(ray, quad) )
    {
        return detail::intersect_mt_bl_uv(ray, quad);
    }
};

struct quad_intersector_pluecker : basic_intersector<quad_intersector_pluecker>
{
    using basic_intersector<quad_intersector_pluecker>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_pluecker(ray, quad) )
    {
        return detail::intersect_pluecker(ray, quad);
    }
};

struct quad_intersector_project_2D : basic_intersector<quad_intersector_project_2D>
{
    using basic_intersector<quad_intersector_project_2D>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_project_2D(ray, quad) )
    {
        return detail::intersect_project_2D(ray, quad);
    }
};

struct quad_intersector_uv : basic_intersector<quad_intersector_uv>
{
    using basic_intersector<quad_intersector_uv>::operator();

    template <typename R, typename S>
    VSNRAY_FUNC
    auto operator()(
            R const& ray,
            basic_quad<S> const& quad
            )
        -> decltype( detail::intersect_uv(ray, quad) )
    {
        return detail::intersect_uv(ray, quad);
    }
};


template <typename HR, typename T>
VSNRAY_FUNC
inline vector<3, T> get_normal(HR const hr, basic_quad<T> const& quad)
{
    VSNRAY_UNUSED(hr);
    return normalize( cross(quad.v2 - quad.v1, quad.v3 - quad.v1) );
}


template <typename TexCoords, typename R, typename T>
VSNRAY_FUNC
inline auto get_tex_coord(
        TexCoords                                   tex_coords,
        hit_record<R, primitive<unsigned>> const&   hr,
        basic_quad<T>                               /* */
        )
    -> typename std::iterator_traits<TexCoords>::value_type
{
    auto t1 = tex_coords[hr.prim_id * 4];
    auto t2 = tex_coords[hr.prim_id * 4 + 1];
    auto t3 = tex_coords[hr.prim_id * 4 + 2];
    auto t4 = tex_coords[hr.prim_id * 4 + 3];

    auto t11 = lerp(t1, t2, hr.u);
    auto t12 = lerp(t3, t4, hr.u);

    return lerp(t11, t12, hr.v);
}

} // visionaray
