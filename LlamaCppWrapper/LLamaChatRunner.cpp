#include "./LLamaChatRunner.h"

#if __has_include (<llama.h>)

#include <iostream>

#include "./Utils/Logger.h"


#pragma comment(lib, "llama.lib")
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "ggml-base.lib")


//https://towardsdatascience.com/llama-cpp-writing-a-simple-c-inference-program-for-gguf-llm-models-12bc5f58505f

#include <gguf.h>

LLamaChatRunner::LLamaChatRunner(const std::string& modelPath, const Settings& sets) :
    LLamaWrapper(modelPath, sets),
    promptBuilderCallback(nullptr)
{
        
}

LLamaChatRunner::~LLamaChatRunner()
{
    this->Relase();
}

void LLamaChatRunner::Relase()
{
    this->ClearMessages();

    LLamaWrapper::Relase();
}

void LLamaChatRunner::ClearMessages()
{
    for (auto& c : chats)
    {
        c->Clear();
    }

    this->ClearContext();
}

void LLamaChatRunner::InitModel()
{
    LLamaWrapper::InitModel();

    auto chatTemplate = llama_model_chat_template(model, /* name */ nullptr);
    if (chatTemplate)
    {
        this->modelChatTemplate = chatTemplate;
    }

    this->InitChats();
}

void LLamaChatRunner::InitChats()
{
    this->chats = std::vector<std::shared_ptr<LLamaChat>>(sets.n_parallel, nullptr);

    for (int i = 0; i < sets.n_parallel; i++)
    {
        chats[i] = std::shared_ptr<LLamaChat>(new LLamaChat(this));
    }
    
    this->ClearMessages();
}

const std::string& LLamaChatRunner::GetTemplate() const
{
    return this->modelChatTemplate;
}

void LLamaChatRunner::SetCustomPromptBuilder(CustomPromptBuider callback)
{
    this->promptBuilderCallback = callback;
}

std::shared_ptr<LLamaChat> LLamaChatRunner::GetChat(int index)
{
    return this->chats[index];
}


void LLamaChatRunner::Run()
{
    if (sets.n_parallel == 1)
    {
        auto prompt = this->chats[0]->BuildPrompt();

        std::string response = this->Loop(prompt);

        this->chats[0]->AddResponse(response);
    }
    else 
    {
        std::vector<std::string> prompts(sets.n_parallel);
        for (size_t i = 0; i < prompts.size(); i++)
        {
            prompts[i] = this->chats[i]->BuildPrompt();
        }

        auto responses = this->Loop(prompts);

        size_t c = std::min(prompts.size(), responses.size());
        if (responses.size() != prompts.size())
        {
            MY_LOG_WARNING("Number of responses != number of prompts (%zu vs %zu)", responses.size(), prompts.size());
        }

        for (size_t i = 0; i < c; i++)
        {
            this->chats[i]->AddResponse(responses[i]);
        }
    }
}

bool LLamaChatRunner::RunEncoderPart(const std::vector<llama_seq_id>& seq_ids)
{    
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch))
    {
        MY_LOG_ERROR("%s : failed to eval\n", __func__);
        return false;
    }

    llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL)
    {
        decoder_start_token_id = llama_vocab_bos(vocab);
    }

    this->ClearBatch();
    this->AddToBatch(decoder_start_token_id, 0, seq_ids, false);

    return true;
}

//=====================================================================================================
// Single processing
//=====================================================================================================


std::string LLamaChatRunner::Loop(const std::string& prompt)
{
    std::string response;

    // tokenize the prompt    
    std::vector<llama_token> prompt_tokens = this->Tokenize(prompt);
    if (prompt_tokens.empty())
    {
        return response;
    }

    // prepare a batch for the prompt
    this->InitBatch(std::max<int32_t>((int32_t)prompt_tokens.size(), 1));

    this->ClearBatch();

    //You restart positions at 0 on every Loop(). 
    //If prompt contains the full input/history, clear the KV cache before filling the batch
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_sampler_reset(sampler_configs[0].sampler);

    for (int32_t i = 0; i < (int32_t)prompt_tokens.size(); i++)
    {
        this->AddToBatch(prompt_tokens[i], i, { 0 }, i == (int32_t)prompt_tokens.size() - 1);
    }

    //If prompt contains only new tokens continuing the cached conversation, 
    // preserve the cache and replace your prompt loop and pos initialization with
    /*
    llama_pos pos = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;

    for (int32_t i = 0; i < (int32_t)prompt_tokens.size(); ++i)
    {
        this->AddToBatch(prompt_tokens[i], pos + i, { 0 }, i == (int32_t)prompt_tokens.size() - 1);
    }
    */

            
    const int n_batch = this->GetMaxTokensForAllBatches();

    const int n_ctx = this->GetContextSize();
    
    int32_t pos = (int32_t)prompt_tokens.size();
    
    while (true)
    {
        
        // check if we have enough space in the context to evaluate this batch                
        llama_memory_t mem = llama_get_memory(ctx);
        int n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
        
        int finalContextSize = n_ctx_used + batchTokensCount;
        if (finalContextSize > n_ctx)
        {
            MY_LOG_ERROR("context size exceeded. Available: %d, Required: %d (already used: %d, required for batch: %d)", 
                n_ctx, finalContextSize, n_ctx_used, batchTokensCount);
            break;
        }
        

        auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (r != 0)
        {
            LLamaWrapper::LogDecodeProcessError(r, __func__);
        }

        // sample the next token
        llama_token new_token_id = llama_sampler_sample(sampler_configs[0].sampler, ctx, batchTokensCount - 1);

        // is it an end of generation?
        if (llama_vocab_is_eog(vocab, new_token_id))
        {
            break;
        }

        // convert the token to a string, print it and add it to the response        
        response += this->TokenToString(new_token_id);

        // prepare the next batch with the sampled token
        //batch = llama_batch_get_one(&new_token_id, 1);
        
        this->ClearBatch();
        this->AddToBatch(new_token_id, pos++, { 0 }, true);        
    }

    return response;
}

//=====================================================================================================
// Parallel processing
//=====================================================================================================

std::vector<LLamaChatRunner::PromptInfo> LLamaChatRunner::InitPrompts(const std::vector<std::string>& prompts)
{
    std::vector<LLamaChatRunner::PromptInfo> pi(prompts.size());

    std::vector<std::vector<llama_token>> prompts_tokens(prompts.size());

    int32_t max_prompt_len = 0;

    for (size_t i = 0; i < prompts.size(); i++)
    {
        prompts_tokens[i] = this->Tokenize(prompts[i]);
        pi[i].tokensCount = static_cast<int32_t>(prompts_tokens[i].size());

        max_prompt_len = std::max(max_prompt_len, pi[i].tokensCount);              
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    for (int32_t s = 0; s < sets.n_parallel; ++s)
    {
        llama_sampler_reset(sampler_configs[s].sampler);
    }


    this->InitBatch(max_prompt_len);
           
    // ==== prefill all prompts (batched, interleaved by position) ====
    for (int pos = 0; pos < max_prompt_len; pos++) 
    {
        this->ClearBatch();

        for (int s = 0; s < sets.n_parallel; s++) 
        {
            if (pos >= prompts_tokens[s].size())
            {
                continue;
            }

            const bool is_last_tok = (pos == prompts_tokens[s].size() - 1);
            
            this->AddToBatch(prompts_tokens[s][pos], pos, {s}, is_last_tok);            
        }

        if (batchTokensCount == 0)
        {
            continue;
        }

        // If your model is encoder-decoder, you'd need per-seq encode handling; omitted for brevity.        
        auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (r != 0)        
        {
            LLamaWrapper::LogDecodeProcessError(r, __func__);
            return {};
        }
    }

    return pi;
}

std::vector<std::string> LLamaChatRunner::Loop(const std::vector<std::string>& prompts)
{
    if (prompts.size() != sets.n_parallel)
    {
        MY_LOG_ERROR("Number of prompts (%d) is not equal to number of parallel processings (%d)",
            sets.n_parallel, prompts.size());
        return {};
    }

    std::vector<PromptInfo> pi = this->InitPrompts(prompts);

    if (pi.size() != sets.n_parallel)
    {
        return {};
    }
    
    
    //==========================================================================

    // main loop

    // we will store the parallel decoded sequences in this vector
    std::vector<std::string> streams(sets.n_parallel, "");

    std::vector<int32_t> i_batch(sets.n_parallel);
    std::vector<int32_t> n_pos(sets.n_parallel);      // current position per seq
    
    for (int32_t i = 0; i < sets.n_parallel; i++)
    {
        n_pos[i] = pi[i].tokensCount;
        i_batch[i] = batchTokensCount - 1;
    }

    
    
    while (true) 
    {
        // prepare the next batch
        this->ClearBatch();

        // sample the next token for each parallel sequence / stream
        for (int32_t i = 0; i < sets.n_parallel; i++) 
        {
            if (i_batch[i] < 0) 
            {
                // the stream has already finished
                continue;
            }

            const llama_token new_token_id = llama_sampler_sample(sampler_configs[i].sampler, ctx, i_batch[i]);

            // is it an end of generation? -> mark the stream as finished
            if (llama_vocab_is_eog(vocab, new_token_id)) 
            {
                i_batch[i] = -1;
                continue;
            }
            
            streams[i] += this->TokenToString(new_token_id);
            
            // push this new token for next evaluation
            auto idx = this->AddToBatch(new_token_id, n_pos[i], {i}, true);            

            i_batch[i] = batchTokensCount - 1; //idx
            n_pos[i] += 1;
        }

        // all streams are finished
        if (batchTokensCount == 0)
        {
            break;
        }
        
        // evaluate the current batch with the transformer model
        auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (r != 0)
        {
            LLamaWrapper::LogDecodeProcessError(r, __func__);
            return {};
        }
    }


    std::vector<std::string> response(prompts.size());
    for (int32_t i = 0; i < sets.n_parallel; ++i)
    {
        response[i] += streams[i];        
    }

    return response;
}


//llama_adapter_lora_init

#endif