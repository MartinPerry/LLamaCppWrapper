#ifndef COLOR_SPACE_H
#define COLOR_SPACE_H

template <typename T>
class Image2d;

#include <cstdint>
#include <cmath>
#include <string>
#include <array>
#include <optional>


class ColorSpace 
{
public:

	enum class PixelFormat
	{
		NONE,
		GRAY,
		RGB,
		RGBA		
	};

	static size_t GetChannelsCount(PixelFormat pf);
	
	template <typename T>
	static constexpr PixelFormat GetFormatFromChannelsCount(size_t c)
	{
		if constexpr (std::is_same<T, uint8_t>::value)
		{
			switch (c)
			{
			case 1: return PixelFormat::GRAY;			
			case 3: return PixelFormat::RGB;
			case 4: return PixelFormat::RGBA;
			default: return PixelFormat::NONE;
			}
		}
		else
		{
			switch (c)
			{
			return PixelFormat::NONE;
			}
		}
	};

	
	static std::optional<Image2d<uint8_t>> ConvertToRGB(const Image2d<uint8_t>& input);

	static Image2d<uint8_t> ConvertRgbaToRgb(const Image2d<uint8_t> & input);	
	static Image2d<uint8_t> ConvertGrayToRgb(const Image2d<uint8_t>& input);		
	static Image2d<uint8_t> ConvertRgbToGray(const Image2d<uint8_t> & input);
	static Image2d<uint8_t> ConvertRgbToRgba(const Image2d<uint8_t>& input, uint8_t a);
	
	

	/// <summary>
	/// Special helper method to get maximal pixel intensity
	/// value for gray
	/// uint8 -> 255
	/// float -> 1.0
	/// </summary>
	/// <typeparam name="T"></typeparam>
	/// <returns></returns>
	template <typename T>
	static T GetMaxGrayValue()
	{
		if constexpr (std::is_same<T, uint8_t>::value)
		{
			return 255;
		}

		if constexpr (std::is_same<T, float>::value)
		{
			return 1.0;
		}
	}
};

#endif

