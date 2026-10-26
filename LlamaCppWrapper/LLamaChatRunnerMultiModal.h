#ifndef LLAMA_CHAT_RUNNER_MULTIMODAL_H
#define LLAMA_CHAT_RUNNER_MULTIMODAL_H

#if __has_include (<llama.h>) && __has_include (<mtmd.h>)

#include <vector>
#include <list>
#include <string>

#include <llama.h>
#include <llama-cpp.h>

#include <mtmd.h>

#include "./LLamaChatRunner.h"
#include "./LLamaChat.h"

class LLamaChatRunnerMultiModal : public LLamaChatRunner
{
public:
	
	LLamaChatRunnerMultiModal(const std::string& modelPath, const std::string& projectorPath,
		const Settings& sets);
	virtual ~LLamaChatRunnerMultiModal();
	
	void InitModel() override;

	

protected:
	std::string projectorPath;

	mtmd_context* ctx_vision;
	
	std::vector<llama_pos> n_past_seq;
	
	void InitChats() override;

	bool LoadImageModel();

	
	void EvalMtmd(llama_seq_id seqId, const std::string& prompt, mtmd::bitmaps& bmps);

	virtual std::string Loop(const std::string& prompt) override;
	virtual std::vector<std::string> Loop(const std::vector<std::string>& prompts) override;
};


#endif

#endif