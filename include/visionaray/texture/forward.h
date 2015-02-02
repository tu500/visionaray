// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_TEXTURE_FORWARD_H
#define VSNRAY_TEXTURE_FORWARD_H

#include <cstddef>

namespace visionaray
{

//--------------------------------------------------------------------------------------------------
// Declarations
//

enum tex_address_mode
{
    Wrap = 0,
    Mirror,
    Clamp,
    Border
};


enum tex_filter_mode
{
    Nearest = 0,
    Linear,
    BSpline,
    BSplineInterpol,
    CardinalSpline
};

enum tex_read_mode
{
    ElementType,
    NormalizedFloat
};


template
<
    typename VoxelT,
    tex_read_mode ReadMode,
    size_t Dim
>
class texture;

} // visionaray


#endif // VSNRAY_TEXTURE_FORWARD_H

