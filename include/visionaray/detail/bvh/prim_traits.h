// This file is distributed under the MIT license.
// See the LICENSE file for details.

#ifndef VSNRAY_DETAIL_BVH_PRIM_TRAITS_H
#define VSNRAY_DETAIL_BVH_PRIM_TRAITS_H 1

#pragma once

namespace visionaray
{

template <typename BVH>
struct num_vertices<BVH, typename std::enable_if<is_any_bvh<BVH>::value>::type>
    : num_vertices<BVH>
{
};

template <typename BVH, typename NormalBinding>
struct num_normals<BVH, NormalBinding, typename std::enable_if<is_any_bvh<BVH>::value>::type>
    : num_normals<typename BVH::primitive_type, NormalBinding>
{
};

template <typename BVH>
struct num_tex_coords<BVH, typename std::enable_if<is_any_bvh<BVH>::value>::type>
    : num_tex_coords<BVH>
{
};

} // visionaray

#endif // VSNRAY_DETAIL_BVH_PRIM_TRAITS_H
