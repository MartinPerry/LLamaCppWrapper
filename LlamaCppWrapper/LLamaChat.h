#ifndef LLAMA_CHAT_H
#define LLAMA_CHAT_H

#if __has_include (<llama.h>)

class LLamaChatRunner;

#include <vector>
#include <list>
#include <string>
#include <optional>

#include <llama.h>
#include <llama-cpp.h>


//================================================

class LLamaChat
{
public:

	inline static const char* ROLE_USER = "user";
	inline static const char* ROLE_SYSTEM = "system";	
	inline static const char* ROLE_ASSISTANT = "assistant";	

	enum class ChatRole
	{
		user, //This role represents the human user in the conversation. 
		//Inputs from the user guide the conversation and prompt responses from the assistant.
		//Use whenever the human user is making a statement or asking a question. This is the most frequent role used in a standard interaction.

		system, //The System role is used to provide setup information or context that informs the behavior of the model.
		//This can include instructions or guidelines on how the conversation should proceed.
		//You use this role to set the stage for the interaction.
		//For example, if you want the model to maintain a formal tone throughout the conversation or if you need to specify rules like avoiding certain topics.

		assistant //This is the role of the model itself, responding to user inputs based on the context set by the system.
				  //This role is automatically assumed by the model when it replies to the user’s queries or follows the system’s instructions.
	};
	
	virtual ~LLamaChat();

	void Relase();	
	virtual void Clear();

	const char* GetLastMessage() const;

	void SetSingleQueryMode(bool val);

	void AddMessage(const std::string& message, ChatRole role);
	void AddMessage(const std::string& message);

	void AddResponse(const std::string& message);
		
	friend class LLamaChatRunner;

protected:
		
	bool singleQueryMode;

	// container to store user/assistant messages in the chat
	std::vector<llama_chat_message> messages;

	// stores the string generated after applying
	// the chat-template to all messages in `messages`
	std::vector<char> formattedMessages;

	int prevLen;

	LLamaChatRunner* parent;

	LLamaChat(LLamaChatRunner* parent);

	int ApplyTemplate(std::vector<char>& formattedMessages);
	
	std::string BuildPrompt();
};


#endif

#endif
