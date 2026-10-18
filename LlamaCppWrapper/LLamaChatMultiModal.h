#ifndef LLAMA_CHAT_MULTIMODAL_H
#define LLAMA_CHAT_MULTIMODAL_H

#if __has_include (<llama.h>)

class LLamaChatRunnerMultiModal;

#include <llama.h>
#include <llama-cpp.h>

#include <mtmd.h>

#include "./Utils/Image2d.h"


#include "./LLamaChat.h"

class LLamaChatMultiModal : public LLamaChat
{
public:
	using LLamaChat::AddMessage;

	~LLamaChatMultiModal();

	void Clear() override;

	void AddMessage(const std::string& message,
		const Image2d<uint8_t>& img, LLamaChat::ChatRole role);

	void AddMessage(const std::string& message,
		const Image2d<uint8_t>& img);

	friend class LLamaChatRunnerMultiModal;

protected:

	mtmd::bitmaps bitmaps;

	LLamaChatMultiModal(LLamaChatRunnerMultiModal* parent);

	void ClearImages();

	bool SetImage(const Image2d<uint8_t>& img);
};

#endif

#endif

