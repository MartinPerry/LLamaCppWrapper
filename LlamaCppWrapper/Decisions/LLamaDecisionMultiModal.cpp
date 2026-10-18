#include "./LLamaDecisionMultiModal.h"

#if __has_include(<llama.h>) && __has_include(<mtmd.h>)

#include <llama-ext.h>
#include <algorithm>
#include <cmath>
#include <limits>

#include "../Utils/Logger.h"
#include "../Utils/ColorSpace.h"

#include "./LLamaDecisionContext.h"

#ifdef _MSC_VER
#pragma comment(lib, "mtmd.lib")
#endif

LLamaDecisionMultiModal::LLamaDecisionMultiModal(const std::string& modelPath,
    const std::string& projectorPath, const Settings& settings) :
    LLamaDecision(modelPath, settings), 
    projectorPath(projectorPath)
{   
    LoadMediaModel();
}

LLamaDecisionMultiModal::~LLamaDecisionMultiModal()
{
}

void LLamaDecisionMultiModal::InitModel()
{
    ctx_vision.reset();
    LLamaDecision::InitModel();
    LoadMediaModel();
}

void LLamaDecisionMultiModal::Relase()
{
    ctx_vision.reset(); // MTMD borrows the text model: release it first.
    dcx.reset();
    LLamaWrapper::Relase();
}

bool LLamaDecisionMultiModal::LoadMediaModel()
{
    if ((dcx->SupportsMedia() == false) || (projectorPath.empty())) 
    {
        MY_LOG_ERROR("a supported decision model and projector path are required for media");
        return false;
    }

    if (!llama_model_has_decoder(model)) 
    {
        MY_LOG_ERROR("decision media require a decoder model");
        return false;
    }

    auto mparams = mtmd_context_params_default();
    mparams.use_gpu = sets.gpuLayers > 0;
    mparams.print_timings = sets.logEnabled;
    mparams.n_threads = 1;
    mtmd_log_set(&LLamaWrapper::CustomErrorLog, &sets);

    auto mtmdModelCtx = mtmd_init_from_file(this->projectorPath.c_str(), model, mparams);

    ctx_vision.reset(mtmdModelCtx);
    
    if (!ctx_vision) 
    {
        MY_LOG_ERROR("failed to load decision projector from %s", projectorPath.c_str());
        return false;
    }

    return true;
}

bool LLamaDecisionMultiModal::SupportsVision() const
{
    return mtmd_support_vision(ctx_vision.get());
}

bool LLamaDecisionMultiModal::SupportsAudio() const
{
    return mtmd_support_audio(ctx_vision.get());
}

int LLamaDecisionMultiModal::GetAudioSampleRate() const
{
    return mtmd_get_audio_sample_rate(ctx_vision.get());
}

std::unordered_map<std::string, DecisionAnswer> LLamaDecisionMultiModal::Run(
    const std::vector<Question>& questions, const std::vector<DecisionMedia>& media)
{
    return Run("", questions, media);
}

std::unordered_map<std::string, DecisionAnswer> LLamaDecisionMultiModal::Run(const std::string& state,
    const std::vector<Question>& questions, const std::vector<DecisionMedia>& media)
{
    if (media.empty())
    {
        return LLamaDecision::Run(state, questions);
    }
    
    using Bitmap = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;

    std::vector<Bitmap> owned;
    std::vector<const mtmd_bitmap*> bitmaps;
    DecisionMediaInput input;
    const char* marker = mtmd_get_marker(ctx_vision.get());
    if (!marker || !*marker) 
    {
        MY_LOG_ERROR("decision projector has no media marker");
        return {};
    }

    for (const auto& item : media) 
    {
        Bitmap bitmap(nullptr, mtmd_bitmap_free);

        if (const auto* image = std::get_if<Image2d<uint8_t>>(&item))
        {        
            if (SupportsVision() == false)
            {
                MY_LOG_ERROR("Unsupported decision image");
                return {};
            }

            mtmd_bitmap* result = nullptr;
            if (image->GetChannelsCount() == 1)
            {
                auto imgRgb = ColorSpace::ConvertGrayToRgb(*image);
                result = mtmd_bitmap_init(imgRgb.GetWidth(), imgRgb.GetHeight(), imgRgb.GetData().data());

            }
            else if (image->GetChannelsCount() == 4)
            {
                auto imgRgb = ColorSpace::ConvertRgbaToRgb(*image);
                result = mtmd_bitmap_init(imgRgb.GetWidth(), imgRgb.GetHeight(), imgRgb.GetData().data());

            }
            else
            {
                result = mtmd_bitmap_init(image->GetWidth(), image->GetHeight(), image->GetData().data());
            }

            bitmap.reset(result);            
        } 
        else 
        {
            const auto& audio = std::get<DecisionAudio>(item);
            if (!SupportsAudio() || audio.samples.empty() || audio.sampleRate != GetAudioSampleRate() ||
                audio.samples.size() > std::numeric_limits<uint32_t>::max() ||
                std::any_of(audio.samples.begin(), audio.samples.end(), [](float v) { 
                    return !std::isfinite(v); 
                })) 
            {
                MY_LOG_ERROR("decision audio must be mono PCM F32 at the projector's sample rate");
                return {};
            }

            input.audio = true;
            bitmap.reset(mtmd_bitmap_init_from_audio(audio.samples.size(), audio.samples.data()));
        }

        if (!bitmap) 
        {
            MY_LOG_ERROR("failed to create decision media bitmap");
            return {};
        }

        bitmaps.push_back(bitmap.get());
        owned.push_back(std::move(bitmap));
        input.markers.emplace_back(marker);
    }

    input.tokenize = [&](const std::string& prompt, DecisionPrompt& task) {
        return TokenizeMedia(prompt, bitmaps, task);
    };

    return RunInternal(state, questions, &input);
}

bool LLamaDecisionMultiModal::TokenizeMedia(const std::string& prompt,
    const std::vector<const mtmd_bitmap*>& bitmaps, DecisionPrompt& task) const
{
    task = {};
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(), mtmd_input_chunks_free
    );

    // Match process_mtmd_prompt's default: MTMD adds the model's initial special tokens.
    const mtmd_input_text text{prompt.c_str(), prompt.size(), true, true};

    if (!chunks || mtmd_tokenize(ctx_vision.get(), chunks.get(), &text, bitmaps.data(), bitmaps.size()) != 0) 
    {
        MY_LOG_ERROR("failed to tokenize decision media; check template media markers");
        return false;
    }

    for (size_t i = 0; i < mtmd_input_chunks_size(chunks.get()); ++i) 
    {
        const auto* chunk = mtmd_input_chunks_get(chunks.get(), i);
        const size_t count = mtmd_input_chunk_get_n_tokens(chunk);

        if (count > static_cast<size_t>(llama_n_ctx_seq(ctx)) - task.tokens.size()) 
        {
            MY_LOG_ERROR("decision media exceed the context size");
            task = {};
            return false;
        }

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

            task.mediaChunks.push_back({task.tokens.size(), std::move(copy)});
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

bool LLamaDecisionMultiModal::ProcessPrompt(DecisionPrompt& prompt)
{
    if (prompt.mediaChunks.empty())
    {
        return LLamaDecision::ProcessPrompt(prompt);
    }

    const size_t batchSize = llama_n_batch(ctx);
    const size_t physicalSize = llama_n_ubatch(ctx);

    if (!ctx_vision || !batchSize || !physicalSize || prompt.tokens.size() > llama_n_ctx_seq(ctx) ||
        prompt.tokens.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) 
    {
        MY_LOG_ERROR("multimodal decision does not fit in the context");
        return false;
    }

    size_t tailStart = 0;
    for (const auto& media : prompt.mediaChunks) 
    {
        const size_t count = mtmd_input_chunk_get_n_tokens(media.chunk.get());
        if (media.offset < tailStart || media.offset > prompt.tokens.size() || count == 0 ||
            count > prompt.tokens.size() - media.offset ||
            (mtmd_decode_use_non_causal(ctx_vision.get(), media.chunk.get()) &&
                (count > batchSize || count > physicalSize))) 
        {
            MY_LOG_ERROR("invalid media layout or insufficient batch size for non-causal media");
            return false;
        }

        tailStart = media.offset + count;
    }

    const size_t tailSize = prompt.tokens.size() - tailStart;

    if (tailSize > batchSize || (!prompt.order.empty() && tailSize > physicalSize) ||
        (prompt.labels.empty() && tailSize == 0))
    {
        MY_LOG_ERROR("decision head does not fit in one batch; increase context/batch settings");
        return false;
    }

    for (const auto marker : prompt.markers) 
    {
        if (marker < 0 || static_cast<size_t>(marker) < tailStart) 
        {
            MY_LOG_ERROR("decision output must follow the media prefix");
            return false;
        }
    }

    if (prompt.pointer >= 0 && static_cast<size_t>(prompt.pointer) < tailStart)
    {
        MY_LOG_ERROR("decision pointer must follow the media prefix");
        return false;
    }

    if (!prompt.order.empty() && (prompt.order.size() != prompt.tokens.size() ||
        std::any_of(prompt.order.begin(), prompt.order.begin() + tailStart,
            [](int32_t order) { return order != LLAMA_DECISION_ORDER_NONE; }))) 
    {
        MY_LOG_ERROR("joint decision spans must follow the media prefix");
        return false;
    }

    const bool causal = llama_get_causal_attn(ctx);
    struct RestoreAttention {
        llama_context* context;
        bool causal;
        ~RestoreAttention() { llama_set_causal_attn(context, causal); }
    } restoreAttention{ctx, causal};

    this->InitBatch(static_cast<int32_t>(batchSize));
    if (!batch)
    {
        MY_LOG_ERROR("failed to initialize multimodal decision batch");
        return false;
    }

    llama_pos position = 0;
    size_t offset = 0;
    size_t lastBatchSize = 0;
    auto process = [&]() {
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) 
        {
            MY_LOG_ERROR("multimodal decision evaluation failed");
            return false;
        }
        return true;
    };

    auto addText = [&](size_t end, bool final) {
        while (offset < end) {
            this->ClearBatch();

            llama_set_causal_attn(ctx, causal);
            const size_t count = final ? end - offset : std::min(batchSize, end - offset);
            for (size_t i = 0; i < count; ++i) 
            {
                const int32_t index = llama_batch_ext_add_token(batch, 0, prompt.tokens[offset + i]);
                const llama_pos pos = position + static_cast<llama_pos>(i);
                const bool output = final && (prompt.labels.empty() || offset + i + 1 == end);
                if (index < 0 || !llama_batch_ext_set_pos(batch, index, &pos) ||
                    !llama_batch_ext_set_output_logits(batch, index, output) ||
                    (!prompt.order.empty() && !llama_batch_ext_set_decision_order(batch, index,
                        static_cast<llama_decision_order>(prompt.order[offset + i])))) 
                {
                    MY_LOG_ERROR("failed to add multimodal decision text to the batch");
                    return false;
                }
            }

            if (!process())
            {
                return false;
            }

            position += static_cast<llama_pos>(count);
            offset += count;
            lastBatchSize = count;
        }
        return true;
    };

    const int32_t embeddingSize = llama_model_n_embd(model);
    const int32_t inputSize = llama_model_n_embd_inp(model);

    if (embeddingSize <= 0 || inputSize < embeddingSize || inputSize % embeddingSize != 0) 
    {
        MY_LOG_ERROR("invalid multimodal decision embedding dimensions");
        return false;
    }

    for (const auto& media : prompt.mediaChunks)
    {
        if (!addText(media.offset, false))
        {
            return false;
        }

        const auto* chunk = media.chunk.get();
        const size_t count = mtmd_input_chunk_get_n_tokens(chunk);
        if (mtmd_encode_chunk(ctx_vision.get(), chunk) != 0) 
        {
            MY_LOG_ERROR("failed to encode decision media");
            return false;
        }
        const float* embeddings = mtmd_get_output_embd(ctx_vision.get());
        if (!embeddings) 
        {
            MY_LOG_ERROR("decision media encoder returned no embeddings");
            return false;
        }

        const bool nonCausal = mtmd_decode_use_non_causal(ctx_vision.get(), chunk);
        llama_set_causal_attn(ctx, nonCausal ? false : causal);
        const auto* image = mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_IMAGE
            ? mtmd_input_chunk_get_tokens_image(chunk) : nullptr;

        for (size_t first = 0; first < count;)
        {
            this->ClearBatch();

            const size_t size = std::min(batchSize, count - first);
            for (size_t i = first; i < first + size; ++i) 
            {
                const float* row = embeddings + i * static_cast<size_t>(inputSize);
                const int32_t index = llama_batch_ext_add_embd(batch, 0,
                    {row, 1, static_cast<size_t>(embeddingSize)}
                );

                const llama_pos sequential = position + static_cast<llama_pos>(i);
                llama_pos pos[4] = {sequential, sequential, sequential, sequential};

                if (image && mtmd_decode_use_mrope(ctx_vision.get())) 
                {
                    const auto decoder = mtmd_image_tokens_get_decoder_pos(image, position, i);
                    pos[0] = static_cast<llama_pos>(decoder.t);
                    pos[1] = static_cast<llama_pos>(decoder.y);
                    pos[2] = static_cast<llama_pos>(decoder.x);
                    pos[3] = static_cast<llama_pos>(decoder.z);
                }

                const bool output = media.offset + i + 1 == prompt.tokens.size();
                if (index < 0 || !llama_batch_ext_set_pos(batch, index, pos) ||
                    !llama_batch_ext_set_output_logits(batch, index, output) ||
                    (inputSize > embeddingSize && !llama_batch_ext_set_embd_state(batch, index,
                        {row + embeddingSize, static_cast<size_t>(inputSize / embeddingSize - 1),
                            static_cast<size_t>(embeddingSize)})))
                {
                    MY_LOG_ERROR("failed to add decision media embeddings to the batch");
                    return false;
                }
            }

            if (!process())
            {
                return false;
            }

            first += size;
            lastBatchSize = size;
        }

        position += mtmd_input_chunk_get_n_pos(chunk);
        offset += count;
    }

    if (!addText(prompt.tokens.size(), true))
    {
        return false;
    }

    // Output APIs index the last batch, not the full prompt or its temporal positions.
    for (auto& marker : prompt.markers)
    {
        marker -= static_cast<int32_t>(tailStart);
    }

    if (prompt.pointer >= 0)
    {
        prompt.pointer -= static_cast<int32_t>(tailStart);
    }

    if (!prompt.order.empty())
    {
        prompt.order.erase(prompt.order.begin(), prompt.order.begin() + tailStart);
    }

    prompt.tokens.erase(prompt.tokens.begin(), prompt.tokens.begin() + tailStart);
   
    if (prompt.tokens.empty())
    {
        prompt.tokens.resize(lastBatchSize, LLAMA_TOKEN_NULL);
    }

    prompt.mediaChunks.clear();

    return true;
}

#endif
