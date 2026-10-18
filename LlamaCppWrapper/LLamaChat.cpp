#include "./LLamaChat.h"

#if __has_include (<llama.h>)

#include <iostream>

#include "./Utils/Logger.h"

//#include "../../Utils/Strings/StringUtils.h"

#pragma comment(lib, "llama.lib")
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "ggml-base.lib")



//https://towardsdatascience.com/llama-cpp-writing-a-simple-c-inference-program-for-gguf-llm-models-12bc5f58505f

#include "./LLamaChatRunner.h"

LLamaChat::LLamaChat(LLamaChatRunner* parent) :
    singleQueryMode(true),
    prevLen(0),
    parent(parent)
{
    
    this->Clear();
}

LLamaChat::~LLamaChat()
{
    this->Relase();
}

void LLamaChat::Relase()
{
    this->Clear();
}

void LLamaChat::Clear()
{
    for (auto& msg : messages)
    {
        free(const_cast<char*>(msg.content));
    }

    messages.clear();

    const int n_ctx = parent->GetContextSize();
    formattedMessages = std::vector<char>(n_ctx);
    prevLen = 0;
    
    parent->ClearContext();
}

const char* LLamaChat::GetLastMessage() const
{
    if (this->messages.size() == 0)
    {
        return nullptr;
    }

    return this->messages.back().content;
}

void LLamaChat::SetSingleQueryMode(bool val)
{
    this->singleQueryMode = val;
}

void LLamaChat::AddMessage(const std::string& message, ChatRole role)
{
    auto msg = llama_chat_message();
    msg.content = strdup(message.c_str());
    switch (role)
    {
    case ChatRole::system:
        msg.role = LLamaChat::ROLE_SYSTEM;
        break;
    case ChatRole::assistant:
        msg.role = LLamaChat::ROLE_ASSISTANT;
        break;
    default:
        msg.role = LLamaChat::ROLE_USER;
        break;
    }


    messages.push_back(msg);
}

void LLamaChat::AddMessage(const std::string& message)
{
    this->AddMessage(message, ChatRole::user);
}

void LLamaChat::AddResponse(const std::string& message)
{
    // add the response to the messages
    this->AddMessage(message, ChatRole::assistant);

    if (singleQueryMode == false)
    {
        prevLen = llama_chat_apply_template(
            parent->modelChatTemplate.empty() ? nullptr : parent->modelChatTemplate.c_str(),
            messages.data(),
            messages.size(),
            false,
            nullptr,
            0);

        if (prevLen < 0)
        {
            fprintf(stderr, "failed to apply the chat template\n");
        }
    }
}


//=============================================================================================
//=============================================================================================
//=============================================================================================

int LLamaChat::ApplyTemplate(std::vector<char>& formattedMessages)
{        
    /*
    common_chat_template tmp("", "", "");
    generation_params gpm;
    
    auto res = common_chat_params_init_gemma4(tmp, gpm);
    */
        
    // apply the chat-template 
    int newLen = llama_chat_apply_template(
        parent->modelChatTemplate.empty() ? nullptr : parent->modelChatTemplate.c_str(),
        messages.data(),
        static_cast<int32_t>(messages.size()),
        true,
        formattedMessages.data(),
        static_cast<int32_t>(formattedMessages.size())
    );
    
    return newLen;
}

std::string LLamaChat::BuildPrompt()
{
    if (this->parent->promptBuilderCallback)
    {
        return this->parent->promptBuilderCallback(messages);
    }
   
    // apply the chat-template 
    int new_len = this->ApplyTemplate(formattedMessages);

    if (new_len > (int)formattedMessages.size())
    {
        // resize the output buffer `formattedMessages`
        // and re-apply the chat template

        formattedMessages.resize(new_len);
        new_len = this->ApplyTemplate(formattedMessages);
    }

    if (new_len < 0)
    {
        throw std::runtime_error("failed to apply the chat template");
    }

    return std::string(formattedMessages.begin() + prevLen, formattedMessages.begin() + new_len);
}


#endif