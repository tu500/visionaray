// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_DETAIL_MATH_H
#define VSNRAY_DETAIL_MATH_H

#include <cmath>

#include "../config.h"


namespace MATH_NAMESPACE
{

//--------------------------------------------------------------------------------------------------
// Import required math functions from the standard library.
// Enable ADL!
//

using std::sqrt;

template <typename T>
MATH_FUNC
inline T min(T const& x, T const& y)
{
    return x < y ? x : y;
}

template <typename T>
MATH_FUNC
inline T max(T const& x, T const& y)
{
    return x < y ? y : x;
}


//--------------------------------------------------------------------------------------------------
// Implement some (useful) functions not defined in <cmath>
//

template <typename T>
MATH_FUNC
inline T clamp(T const& x, T const& a, T const& b)
{
    return max( a, min(x, b) );
}

template <typename T>
MATH_FUNC
inline T lerp(T a, T b, T x)
{
    return a + x * (b - a);
}

template <typename T>
MATH_FUNC
inline T rsqrt(T const& x)
{
    return T(1.0) / sqrt(x);
}

template <typename T>
MATH_FUNC
inline T cot(T x)
{
    return T(1.0) / tan(x);
}

template <typename T>
MATH_FUNC
inline T det2(T const& m00, T const& m01, T const& m10, T const& m11)
{
    return m00 * m11 - m10 * m01;
}


//--------------------------------------------------------------------------------------------------
// Constants
//

namespace constants
{
template <typename T> MATH_FUNC T degrees_to_radians()  { return T(1.74532925199432957692369076849e-02); }
template <typename T> MATH_FUNC T radians_to_degrees()  { return T(5.72957795130823208767981548141e+01); }
template <typename T> MATH_FUNC T e()                   { return T(2.71828182845904523536028747135e+00); }
template <typename T> MATH_FUNC T log2_e()              { return T(1.44269504088896338700465094007e+00); }
template <typename T> MATH_FUNC T pi()                  { return T(3.14159265358979323846264338328e+00); }
template <typename T> MATH_FUNC T inv_pi()              { return T(3.18309886183790691216444201928e-01); }
} // constants


//--------------------------------------------------------------------------------------------------
// SIMD function wrappers
//

template <typename T, typename M>
MATH_FUNC
inline T select(M k, T const& a, T const& b)
{
    return k ? a : b;
}

template <typename T, typename M>
MATH_FUNC
T add(T a, T b, M m, T old = T(0.0))
{
    return select( m, a + b, old );
}

template <typename T, typename M>
MATH_FUNC
T sub(T a, T b, M m, T old = T(0.0))
{
    return select( m, a - b, old );
}

template <typename T, typename M>
MATH_FUNC
T mul(T a, T b, M m, T old = T(0.0))
{
    return select( m, a * b, old );
}

template <typename T, typename M>
MATH_FUNC
T div(T a, T b, M m, T old = T(0.0))
{
    return select( m, a / b, old );
}

MATH_FUNC
inline bool any(bool b)
{
    return b;
}

MATH_FUNC
inline bool all(bool b)
{
    return b;
}

} // MATH_NAMESPACE


#endif // VSNRAY_DETAIL_MATH_H

