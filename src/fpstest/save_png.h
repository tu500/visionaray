// This file is distributed under the MIT license.
// See the LICENSE file for details.

#pragma once

#include <iostream>
#include <ostream>

#include <png.h>

#include <visionaray/math/math.h>

#include <common/cfile.h>

namespace visionaray
{
namespace detail
{

struct png_write_context
{
	png_structp png;
	png_infop info;

	png_write_context()
		: png(nullptr)
		, info(nullptr)
	{
	}

	~png_write_context()
	{
		png_destroy_write_struct(&png, &info);
	}
};

} // detail

static void save_png(
	std::string filename,
	int width,
	int height,
	vec4 const* colors // RGBA
	)
{
	cfile file(filename.c_str(), "w");

	if (!file.good())
	{
		std::cerr << "File open error\n";
		return;
	}

	detail::png_write_context context;

	context.png = png_create_write_struct(
		PNG_LIBPNG_VER_STRING,
		0, // TODO: user data
		0, // TODO: error cb
		0  // TODO: warning cb
		);

	if (context.png == nullptr)
	{
		std::cerr << "Error creating png write struct\n";
		return;
	}

	context.info = png_create_info_struct(context.png);

	if (context.info == nullptr)
	{
		std::cerr << "Error creating png info struct\n";
		return;
	}


	png_init_io(context.png, file.get());

	png_set_IHDR(
		context.png,
		context.info,
		width,
		height,
		8, // bits per color channel
		PNG_COLOR_TYPE_RGB_ALPHA,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_BASE,
		PNG_FILTER_TYPE_BASE
		);

	png_write_info(context.png, context.info);

	std::vector<unsigned char> row(4 * width);

    for (int y = height-1; y >= 0; --y)
	{
		for (int x = 0; x < width; ++x)
		{
			vec4 color = colors[y * width + x];
			row[x * 4]     = color.x * 0xFF;
			row[x * 4 + 1] = color.y * 0xFF;
			row[x * 4 + 2] = color.z * 0xFF;
			row[x * 4 + 3] = color.w * 0xFF;
		}
		png_write_row(context.png, row.data());
	}

	png_write_end(context.png, nullptr);
}

} // visionaray
