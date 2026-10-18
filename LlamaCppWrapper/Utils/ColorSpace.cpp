#include "./ColorSpace.h"

#include <array>
#include <algorithm>

#include "./Image2d.h"



/// <summary>
/// Get number of channels based on PixelFormat
/// </summary>
/// <param name="pf"></param>
/// <returns></returns>
size_t ColorSpace::GetChannelsCount(PixelFormat pf)
{
	switch (pf)
	{
	case PixelFormat::GRAY: return 1;
	case PixelFormat::RGB: return 3;	
	case PixelFormat::RGBA: return 4;	
	default: return 0;
	}
}

//=============================================================================================
// Others to ...
//=============================================================================================


std::optional<Image2d<uint8_t>> ColorSpace::ConvertToRGB(const Image2d<uint8_t>& input)
{
	if (input.GetPixelFormat() == PixelFormat::RGB)
	{
		return input;
	}

	if (input.GetPixelFormat() == ColorSpace::PixelFormat::GRAY)
	{
		return ColorSpace::ConvertGrayToRgb(input);
	}
	else if (input.GetPixelFormat() == ColorSpace::PixelFormat::RGBA)
	{
		return ColorSpace::ConvertRgbaToRgb(input);		
	}

	return std::nullopt;
}

/// <summary>
/// Convert Grayscale image to RGB
/// (Each channel has the same value)
/// </summary>
/// <param name="input"></param>
/// <returns></returns>
Image2d<uint8_t> ColorSpace::ConvertGrayToRgb(const Image2d<uint8_t> & input)
{
	if (input.GetPixelFormat() == PixelFormat::RGB)
	{
		return input;
	}

	size_t len = input.GetPixelsCount();
	std::vector<uint8_t> data;
	data.resize(len * 3);

	size_t outIndex = 0;
	for (size_t i = 0; i < len; i++)
	{
		const uint8_t gray = input.GetData()[i];
		
		data[outIndex++] = gray;
		data[outIndex++] = gray;
		data[outIndex++] = gray;

	}

	return Image2d<uint8_t>(input.GetWidth(),
		input.GetHeight(),
		std::move(data),
		PixelFormat::RGB);
}

Image2d<uint8_t> ColorSpace::ConvertRgbaToRgb(const Image2d<uint8_t> & input)
{
	if (input.GetPixelFormat() == PixelFormat::RGB)
	{
		return input;
	}

	size_t len = input.GetPixelsCount();
	std::vector<uint8_t> data;
	data.resize(len * 3);

	size_t outIndex = 0;
	for (size_t i = 0; i < len; i++)
	{
		const uint8_t * rgba = input.GetPixelStart(i);

		data[outIndex++] = rgba[0];
		data[outIndex++] = rgba[1];
		data[outIndex++] = rgba[2];

	}

	return Image2d<uint8_t>(input.GetWidth(),
		input.GetHeight(),
		std::move(data),
		PixelFormat::RGB);
}


//=============================================================================================
// RGB to ...
//=============================================================================================

/// <summary>
/// Convert RGB to Grayscale
/// </summary>
/// <param name="input"></param>
/// <returns></returns>
Image2d<uint8_t> ColorSpace::ConvertRgbToGray(const Image2d<uint8_t> & input)
{
	if (input.GetPixelFormat() == PixelFormat::GRAY)
	{
		return input;
	}

	size_t len = input.GetPixelsCount();
	std::vector<uint8_t> data;
	data.resize(len);

	size_t len48 = 0;


	for (size_t i = len48; i < len; i++)
	{
		const uint8_t * rgb = input.GetPixelStart(i);

		float gray = rgb[0] * 0.299f + rgb[1] * 0.587f + rgb[2] * 0.114f;

		data[i] = (gray > 255.0f) ? static_cast<uint8_t>(255) :
			(gray < 0.0f) ? static_cast<uint8_t>(0) : static_cast<uint8_t>(gray);		

	}

	return Image2d<uint8_t>(input.GetWidth(),
		input.GetHeight(),
		std::move(data),
		PixelFormat::GRAY);
}

Image2d<uint8_t> ColorSpace::ConvertRgbToRgba(const Image2d<uint8_t>& input, uint8_t a)
{
	if (input.GetPixelFormat() == PixelFormat::RGBA)
	{
		return input;
	}

	size_t len = input.GetPixelsCount();
	std::vector<uint8_t> data;
	data.resize(len * 4);

	size_t index = 0;
	for (size_t i = 0; i < len; i++)
	{
		const uint8_t* rgb = input.GetPixelStart(i);
		
		data[index++] = rgb[0];
		data[index++] = rgb[1];
		data[index++] = rgb[2];
		data[index++] = a;

	}

	return Image2d<uint8_t>(input.GetWidth(),
		input.GetHeight(),
		std::move(data),
		PixelFormat::RGBA);
}

