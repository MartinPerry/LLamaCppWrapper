#ifndef IMAGE_2D_H
#define IMAGE_2D_H


#include <vector>
#include <functional>
#include <optional>

#if __has_include (<opencv2/imgproc.hpp>)
#	include <opencv2/imgproc.hpp>
#	define HAVE_OPENCV 1
#endif

#include "./ColorSpace.h"


/// <summary>
/// Image dimension structure
/// Holds width and height
/// </summary>
struct ImageDimension
{
	int w;
	int h;

	ImageDimension() :
		w(0),
		h(0)
	{
	}

	ImageDimension(int w, int h) :
		w(w),
		h(h)
	{
	}	
};


template <typename T>
class Image2d 
{
public:
		
	Image2d();			
	Image2d(int w, int h, ColorSpace::PixelFormat pf);	
	Image2d(int w, int h, const std::vector<T> & data, ColorSpace::PixelFormat pf);
	Image2d(int w, int h, const T * rawData, ColorSpace::PixelFormat pf);
	Image2d(int w, int h, const T ** rawData, ColorSpace::PixelFormat pf);
	Image2d(int w, int h, std::vector<T> && data, ColorSpace::PixelFormat pf);
	Image2d(const Image2d<T> & other);
	Image2d(Image2d<T> && other) noexcept;
	~Image2d();
	
	void Release() noexcept;

	Image2d<T> & operator=(const Image2d<T> & other);
	Image2d<T> & operator=(Image2d<T> && other) noexcept;

	
	ColorSpace::PixelFormat GetPixelFormat() const noexcept;
	size_t GetChannelsCount() const noexcept;
	

	int GetWidth() const noexcept;
	int GetHeight() const noexcept;
	size_t GetPixelsCount() const noexcept;
	const ImageDimension & GetDimension() const noexcept;
	
	void GetPositionFromIndex(size_t index, int & x, int & y) const noexcept;
	size_t GetIndexFromPosition(int x, int y) const noexcept;
	bool IsEdgePixel(int x, int y) const noexcept;

	const T * GetPixelStart(size_t index) const;
	const T * GetPixelStart(int x, int y) const;
	T * GetPixelStart(size_t index);
	T * GetPixelStart(int x, int y);
	const T * GetRowStart(int y) const;
	T* GetRowStart(int y);
	const T * operator[](size_t index) const;
	T * operator[](size_t index);

	const std::vector<T>& GetData() const noexcept;
	std::vector<T>& GetData() noexcept;
	T* GetRawDataDeepCopy() const;
	std::vector<T>&& MoveData() noexcept;


protected:	

	ImageDimension dim;
	std::vector<T> data;
	ColorSpace::PixelFormat pf;	
	size_t channelsCount;
		
};



#endif
