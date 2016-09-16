#include <sstream>

namespace snex {

namespace detail {

//-------------------------------------------------------------------------------------------------
// intersect by using test with Pluecker coordinates (cf. Shevtsov et al. 2007,
//      "Ray-Triangle Intersection Algorithm for Modern CPU Architectures"
// TODO: implement/test the precalculations proposed in the paper
//

template <typename R>
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

    T s1 = copysign(T(1.0), dot(e1, r));
    T s2 = copysign(T(1.0), dot(e2, r));
    T s3 = copysign(T(1.0), dot(e3, r));
    T s4 = copysign(T(1.0), dot(e4, r));

    result.hit = s1 == s2 && s1 == s3 && s1 == s4;

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
#if 1
        V2 uv = get_uv(quad, result.isect_pos);
        result.u = uv.x;
        result.v = uv.y;

        if (all(result.hit) && any( result.hit && (
                    uv.x < -0.1 || uv.x > 1.1 ||
                    uv.y < -0.1 || uv.y > 1.1
                    )))
        {
            std::stringstream ss;
            ss << result.hit << std::endl;
            ss << uv.x << std::endl;
            ss << uv.y << std::endl;
            std::cout << ss.str() << std::endl;
        }
#endif
    }

    return result;
}

} // detail

} // snex