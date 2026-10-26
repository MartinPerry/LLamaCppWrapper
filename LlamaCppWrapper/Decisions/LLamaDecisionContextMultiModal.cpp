#include "./LLamaDecisionContextMultiModal.h"

#if __has_include(<llama.h>) && __has_include(<mtmd.h>)

#include "../Utils/Logger.h"

bool DecisionContextMultiModal::Tokenize(mtmd_context* ctx_vision,
    const std::string& prompt,
    const std::vector<mtmd_bitmap*>& bitmaps, DecisionPrompt& task)
{
    task = {};
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(), mtmd_input_chunks_free
    );

    // Match process_mtmd_prompt's default: MTMD adds the model's initial special tokens.
    const mtmd_input_text text{ prompt.c_str(), prompt.size(), true, true };

    if (!chunks || mtmd_tokenize(ctx_vision, chunks.get(), &text, bitmaps.data(), bitmaps.size()) != 0)
    {
        MY_LOG_ERROR("failed to tokenize decision media; check template media markers");
        return false;
    }

    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i)
    {
        const auto* chunk = mtmd_input_chunks_get(chunks.get(), i);
        const size_t count = mtmd_input_chunk_get_n_tokens(chunk);

        //todo
        /*
        if (count > static_cast<size_t>(llama_n_ctx_seq(ctx)) - task.tokens.size())
        {
            MY_LOG_ERROR("decision media exceed the context size");
            task = {};
            return false;
        }
        */

        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT)
        {
            size_t textCount = 0;
            const auto* tokens = mtmd_input_chunk_get_tokens_text(chunk, &textCount);
            if (textCount)
            {
                task.tokens.insert(task.tokens.end(), tokens, tokens + textCount);
            }
        }
        else
        {
            auto copy = std::shared_ptr<const mtmd_input_chunk>(mtmd_input_chunk_copy(chunk), mtmd_input_chunk_free);
            if (!copy || count == 0)
            {
                MY_LOG_ERROR("empty decision media chunk");
                task = {};
                return false;
            }

            task.mediaChunks.push_back({ task.tokens.size(), std::move(copy) });
            task.tokens.insert(task.tokens.end(), count, LLAMA_TOKEN_NULL);
        }
    }

    if (task.mediaChunks.empty())
    {
        MY_LOG_ERROR("decision template did not produce media chunks");
        task = {};
        return false;
    }

    return true;
}

DecisionContextMultiModal::DecisionContextMultiModal(DecisionModelType modelType) : 
    DecisionContext(modelType)
{
}

void DecisionContextMultiModal::Init(const llama_model* model, const std::string& prefix)
{
    DecisionContext::Init(model, prefix);
}

std::string DecisionContextMultiModal::RunTemplate(const PromptInput& input) const
{        
    // Escape caller-controlled media markers before adding the real template markers.
    auto clean = [&](std::string& text) {
        for (const auto& marker : mediaInput->markers)
        {
            replace_all(text, marker, " ");
        }
    };
    auto cleanQuestion = [&](Question& question) {
        clean(question.id);
        clean(question.instructions);
        for (auto& option : question.criteria) 
        {
            clean(option.key);
            if (option.value)
            {
                clean(*option.value);
            }
        }
    };


    PromptInput prepared = input;

    clean(prepared.state);
    cleanQuestion(prepared.question);
    
    for (auto& question : prepared.questions)
    {
        cleanQuestion(question);
    }

    for (auto& label : prepared.labels)
    {
        clean(label);
    }

    for (auto& labels : prepared.questionLabels)
    {
        for (auto& label : labels)
        {
            clean(label);
        }
    }

    prepared.images = mediaInput->markers;
    prepared.audio = mediaInput->audio;

    return DecisionContext::RunTemplate(prepared);    
}

bool DecisionContextMultiModal::TokenizePrompt(const std::string& prompt, DecisionPrompt& task) const
{    
    return Tokenize(mediaInput->ctx_vision, prompt, mediaInput->bitmaps, task);    
}

float DecisionContextMultiModal::GetTemperature(const PreparedQuestion& question) const
{
    if (modelType == DecisionModelType::LFM2_D1_OMNI)
    {
        return 1.0f;
    }

    return DecisionContext::GetTemperature(question);    
}

#endif