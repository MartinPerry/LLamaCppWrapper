#ifndef LLAMA_CHAT_RUNNER_H
#define LLAMA_CHAT_RUNNER_H

#if __has_include (<llama.h>)

#include <vector>
#include <list>
#include <string>
#include <optional>
#include <functional>

#include <llama.h>
#include <llama-cpp.h>

#include "./LLamaWrapper.h"
#include "./LLamaChat.h"


//================================================

class LLamaChatRunner : public LLamaWrapper
{
public:
	
	using CustomPromptBuider = std::function<std::string(const std::vector<llama_chat_message>&)>;

	LLamaChatRunner(const std::string& modelPath, const Settings& sets);
	~LLamaChatRunner();

	void Relase() override;
	void ClearMessages();	

	virtual void InitModel() override;

	const std::string& GetTemplate() const;

	void SetCustomPromptBuilder(CustomPromptBuider callback);
	
	std::shared_ptr<LLamaChat> GetChat(int index);

	template <typename ChatType>
	std::shared_ptr<ChatType> GetChatAs(int index)
	{
		return std::dynamic_pointer_cast<ChatType>(this->GetChat(index));
	}
		

	void Run();
	
	friend class LLamaChat;

protected:
	struct PromptInfo 
	{
		int32_t tokensCount;
		//int32_t iBatch;
	};
	
	
	std::string modelChatTemplate;
	CustomPromptBuider promptBuilderCallback;
	
	std::vector<std::shared_ptr<LLamaChat>> chats;

	
	virtual void InitChats();

	bool RunEncoderPart(const std::vector<llama_seq_id>& seq_ids);

	std::vector<PromptInfo> InitPrompts(const std::vector<std::string>& prompts);

	
	virtual std::string Loop(const std::string& prompt);
	virtual std::vector<std::string> Loop(const std::vector<std::string>& prompts);
};


#endif

#endif
