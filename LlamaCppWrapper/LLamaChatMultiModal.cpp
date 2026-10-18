#include "./LLamaChatMultiModal.h"

#if __has_include (<llama.h>)

#include "./Utils/ColorSpace.h"

#include "./LLamaChatRunnerMultiModal.h"
#include "./LLamaChatRunner.h"

LLamaChatMultiModal::LLamaChatMultiModal(LLamaChatRunnerMultiModal* parent) : 
	LLamaChat(dynamic_cast<LLamaChatRunner*>(parent))
{
}

LLamaChatMultiModal::~LLamaChatMultiModal()
{
}

void LLamaChatMultiModal::Clear()
{
	LLamaChat::Clear();

	this->ClearImages();
}

void LLamaChatMultiModal::ClearImages()
{
	this->bitmaps.entries.clear();
}

void LLamaChatMultiModal::AddMessage(const std::string& message, const Image2d<uint8_t>& img, LLamaChat::ChatRole role)
{
	this->SetImage(img);

	if (message.find(mtmd_default_marker()) == std::string::npos)
	{
		std::string mediaMessage = message;
		for (size_t i = 0; i < this->bitmaps.entries.size(); i++)
		{
			mediaMessage += mtmd_default_marker();
		}
		
		this->AddMessage(mediaMessage, role);
	}
	else
	{
		this->AddMessage(message, role);
	}
}


void LLamaChatMultiModal::AddMessage(const std::string& message, const Image2d<uint8_t>& img)
{
	this->AddMessage(message, img, ChatRole::user);
}

bool LLamaChatMultiModal::SetImage(const Image2d<uint8_t>& img)
{
	mtmd_bitmap* result = nullptr;

	if (img.GetChannelsCount() == 1)
	{
		auto imgRgb = ColorSpace::ConvertGrayToRgb(img);
		result = mtmd_bitmap_init(imgRgb.GetWidth(), imgRgb.GetHeight(), imgRgb.GetData().data());
	}
	else
	{
		result = mtmd_bitmap_init(img.GetWidth(), img.GetHeight(), img.GetData().data());
	}

	mtmd::bitmap bmp(result);
	if (!bmp.ptr)
	{
		return false;
	}
	bitmaps.entries.push_back(std::move(bmp));
	return true;
}

#endif