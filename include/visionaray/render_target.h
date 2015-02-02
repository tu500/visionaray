// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#ifndef VSNRAY_RENDER_TARGET_H
#define VSNRAY_RENDER_TARGET_H

#include <cstdint>
#include <vector>

#include "detail/aligned_vector.h"
#include "pixel_format.h"

namespace visionaray
{

class render_target
{
public:

    size_t width() const { return width_; }
    size_t height() const { return height_; }

    void begin_frame()
    {
        begin_frame_impl();
    }

    void end_frame()
    {
        end_frame_impl();
    }

    void resize(size_t w, size_t h)
    {
        width_ = w;
        height_ = h;
        resize_impl(w, h);
    }

    void display_color_buffer() const
    {
        display_color_buffer_impl();
    }

    virtual void* color() = 0;
    virtual void* depth() = 0;

    virtual void const* color() const = 0;
    virtual void const* depth() const = 0;

private:

    size_t width_;
    size_t height_;

    virtual void begin_frame_impl() = 0;
    virtual void end_frame_impl() = 0;
    virtual void resize_impl(size_t w, size_t h) = 0;
    virtual void display_color_buffer_impl() const = 0;

};

class cpu_buffer_rt : public render_target
{
public:

    typedef aligned_vector<uint8_t> buffer_type;

    cpu_buffer_rt(pixel_format cf, pixel_format df);

    void* color();
    void* depth();

    void const* color() const;
    void const* depth() const;

private:

    buffer_type color_buffer_;
    buffer_type depth_buffer_;

    pixel_format color_format_;
    pixel_format depth_format_;

    void begin_frame_impl();
    void end_frame_impl();
    void resize_impl(size_t w, size_t h);
    void display_color_buffer_impl() const;

};

} // visionaray

#endif // VSNRAY_RENDER_TARGET_H

