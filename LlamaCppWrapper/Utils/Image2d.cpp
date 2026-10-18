#include "./Image2d.h"

#include <algorithm>
#include <cstring>


//=================================================================================================
// ctors & dtor
//=================================================================================================

/// <summary>
/// Create empty image with size 0
/// </summary>
template <typename T>
Image2d<T>::Image2d() : 
	dim({ 0, 0 }),
	pf(ColorSpace::PixelFormat::NONE),
	channelsCount(ColorSpace::GetChannelsCount(ColorSpace::PixelFormat::NONE))
{
}


/// <summary>
/// Create empty image with size w x h and given pixel format
/// </summary>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="pf"></param>
template <typename T>
Image2d<T>::Image2d(int w, int h, ColorSpace::PixelFormat pf) :
	dim({ w, h }),	
	data(std::vector<T>(w * h * ColorSpace::GetChannelsCount(pf), 0)),
	pf(pf),
	channelsCount(ColorSpace::GetChannelsCount(pf))
{		
}

/// <summary>
/// Create image with size w x h, filled with data and given pixel format
/// </summary>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="data"></param>
/// <param name="pf"></param>
template <typename T>
Image2d<T>::Image2d(int w, int h, const std::vector<T> & data, ColorSpace::PixelFormat pf) :
	dim({ w, h }),
	data(data),
	pf(pf),
	channelsCount(ColorSpace::GetChannelsCount(pf))
{
}

/// <summary>
/// Create image with size w x h, filled with data and given pixel format
/// </summary>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="data"></param>
/// <param name="pf"></param>
template <typename T>
Image2d<T>::Image2d(int w, int h, const T * rawData, ColorSpace::PixelFormat pf) :
	dim({ w, h }),
	data(rawData, rawData + w * h * ColorSpace::GetChannelsCount(pf)),
	pf(pf),
	channelsCount(ColorSpace::GetChannelsCount(pf))
{
}

/// <summary>
/// Create image with size w x h, filled with data and given pixel format
/// Data are array of arrays 
/// First dimension has size of height, each "entry" has size width
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="rawData"></param>
/// <param name="pf"></param>
template <typename T>
Image2d<T>::Image2d(int w, int h, const T ** rawData, ColorSpace::PixelFormat pf) :
	dim({ w, h }),	
	pf(pf),
	channelsCount(ColorSpace::GetChannelsCount(pf))
{
	for (int i = 0; i < h; i++)
	{
		data.insert(data.end(), rawData[i], rawData[i] + w * channelsCount);
	}
}

/// <summary>
/// Create image with size w x h, filled with data and given pixel format
/// data are moved
/// </summary>
/// <param name="w"></param>
/// <param name="h"></param>
/// <param name="data"></param>
/// <param name="pf"></param>
template <typename T>
Image2d<T>::Image2d(int w, int h, std::vector<T> && data, ColorSpace::PixelFormat pf) :
	dim({ w, h }),	
	data(std::exchange(data, {})),
	pf(pf),
	channelsCount(ColorSpace::GetChannelsCount(pf))
{	
}

template <typename T>
Image2d<T>::Image2d(const Image2d<T> & other) : 
	dim(other.dim),	
	data(other.data),
	pf(other.pf),
	channelsCount(other.channelsCount)
{
}

/// <summary>
/// move ctor
/// </summary>
/// <param name="other"></param>
template <typename T>
Image2d<T>::Image2d(Image2d<T> && other) noexcept :
	dim(other.dim),
	data(std::exchange(other.data, {})),
	pf(other.pf),
	channelsCount(other.channelsCount)
{
	other.Release();
}


/// <summary>
/// dtor
/// </summary>
template <typename T>
Image2d<T>::~Image2d()
{	
}

/// <summary>
/// Manually release data
/// dimensions of image are set to 0
/// </summary>
template <typename T>
void Image2d<T>::Release() noexcept
{
	this->data.clear();
	this->channelsCount = 0;
	this->dim.w = 0;
	this->dim.h = 0;
	this->pf = ColorSpace::PixelFormat::NONE;
}


//=================================================================================================
// Operators section
//=================================================================================================


template <typename T>
Image2d<T> & Image2d<T>::operator=(const Image2d<T> & other)
{
	this->dim = other.dim;	
	this->data = other.data;
	this->pf = other.pf;
	this->channelsCount = other.channelsCount;
	return *this;
}

template <typename T>
Image2d<T> & Image2d<T>::operator=(Image2d<T> && other) noexcept
{
	this->dim = other.dim;	
	this->data = std::exchange(other.data, {});
	this->pf = other.pf;
	this->channelsCount = other.channelsCount;

	other.Release();

	return *this;
}

//=================================================================================================
// Getters
//=================================================================================================


template <typename T>
ColorSpace::PixelFormat Image2d<T>::GetPixelFormat() const noexcept
{
	return this->pf;
}

template <typename T>
size_t Image2d<T>::GetChannelsCount() const noexcept
{
	return this->channelsCount;
}


template <typename T>
int Image2d<T>::GetWidth() const noexcept
{
	return this->dim.w;
}

template <typename T>
int Image2d<T>::GetHeight() const noexcept
{
	return this->dim.h;
}

template <typename T>
size_t Image2d<T>::GetPixelsCount() const noexcept
{
	return size_t(this->dim.w) * size_t(this->dim.h);
}

template <typename T>
const ImageDimension & Image2d<T>::GetDimension() const noexcept
{
	return this->dim;
}

/// <summary>
/// Convert 1D index to 2D [x, y]
/// </summary>
/// <param name="index"></param>
/// <param name="x"></param>
/// <param name="y"></param>
template <typename T>
void Image2d<T>::GetPositionFromIndex(size_t index, int & x, int & y) const noexcept
{
	x = static_cast<int>(index % this->GetWidth());
	y = static_cast<int>(index / this->GetWidth());
}

/// <summary>
/// Convert 2D index [x, y] to 1D index
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
template <typename T>
size_t Image2d<T>::GetIndexFromPosition(int x, int y) const noexcept
{
	return size_t(x) + size_t(y) * size_t(this->dim.w);
}

/// <summary>
/// Check if index [x, y] is on the edge of image
/// (= at least one of its neighbor is outside of image)
/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
template <typename T>
bool Image2d<T>::IsEdgePixel(int x, int y) const noexcept
{
	return ((x == 0) || (x == this->GetWidth() - 1) ||
			(y == 0) || (y == this->GetHeight() - 1));
}


/// <summary>
/// Get pointer to the first element of pixel
/// </summary>
/// <param name="index"></param>
/// <returns></returns>
template <typename T>
const T * Image2d<T>::GetPixelStart(size_t index) const
{
	return &this->data[index * this->GetChannelsCount()];
}

template <typename T>
T * Image2d<T>::GetPixelStart(size_t index)
{
	return &this->data[index * this->GetChannelsCount()];
}



/// <summary>
/// Get pointer to the first byte of pixel
/// </summary>
/// <param name="x"></param>
/// <param name="y"></param> 
/// <returns></returns>
template <typename T>
const T * Image2d<T>::GetPixelStart(int x, int y) const
{
	return this->GetPixelStart(this->GetIndexFromPosition(x, y));
}

template <typename T>
T * Image2d<T>::GetPixelStart(int x, int y)
{
	return this->GetPixelStart(this->GetIndexFromPosition(x, y));
}

template <typename T>
const T* Image2d<T>::GetRowStart(int y) const
{
	return this->GetPixelStart(this->GetIndexFromPosition(0, y));
}

template <typename T>
T* Image2d<T>::GetRowStart(int y)
{
	return this->GetPixelStart(this->GetIndexFromPosition(0, y));
}

template <typename T>
const T * Image2d<T>::operator[](size_t index) const
{
	return this->GetPixelStart(index);
}

template <typename T>
T * Image2d<T>::operator[](size_t index)
{
	return this->GetPixelStart(index);
}

template <typename T>
const std::vector<T>& Image2d<T>::GetData() const noexcept
{
	return this->data;
}

template <typename T>
std::vector<T>& Image2d<T>::GetData() noexcept
{
	return this->data;
}

template <typename T>
T* Image2d<T>::GetRawDataDeepCopy() const
{
	T* rawData = new T[this->data.size()];
	std::copy(this->data.begin(), this->data.end(), rawData);
	return rawData;
}

template <typename T>
std::vector<T>&& Image2d<T>::MoveData() noexcept
{
	return std::move(this->data);
}



template class Image2d<uint8_t>;
template class Image2d<float>;
