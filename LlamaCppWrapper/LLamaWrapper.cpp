#include "./LLamaWrapper.h"

#if __has_include (<llama.h>)

#include <iostream>

#include "./Utils/Logger.h"


#pragma comment(lib, "llama.lib")
#pragma comment(lib, "ggml.lib")
#pragma comment(lib, "ggml-base.lib")


//https://towardsdatascience.com/llama-cpp-writing-a-simple-c-inference-program-for-gguf-llm-models-12bc5f58505f

#include <gguf.h>

void LLamaWrapper::CustomErrorLog(enum ggml_log_level level, const char* text, void* user_data)
{
    const LLamaWrapper::Settings* sets = (const LLamaWrapper::Settings*)(user_data);

    if ((sets->logEnabled) && (level >= GGML_LOG_LEVEL_DEBUG))
    {
        size_t len = strlen(text);
        if (text[len - 1] == '\n')
        {
            MyUtils::Logger::GetInstance()->SetLineEnd("");
        }

        if (level == GGML_LOG_LEVEL_ERROR)
        {
            MY_LOG_ERROR("%s", text);
        }
        else if (level == GGML_LOG_LEVEL_WARN)
        {
            MY_LOG_WARNING("%s", text);
        }
        else if (level == GGML_LOG_LEVEL_INFO)
        {
            MY_LOG_INFO("%s", text);
        }
        else
        {
            MY_LOG("%s", text);
        }

        if (text[len - 1] == '\n')
        {
            //restore
            MyUtils::Logger::GetInstance()->SetLineEnd("\n");
        }
    }
}

LLamaWrapper::LLamaWrapper(const std::string& modelPath, const Settings& sets) :
    modelPath(modelPath),
    sets(sets),
    ctx(nullptr),
    model(nullptr),    
    batch(nullptr),
    batchTokensCount(0),
    currToken(0),
    processedInputTokensCount(0),
    processedOutputTokensCount(0),
    vocab(nullptr)
{
    if (sets.logEnabled == false) 
    {
        // only print errors
        llama_log_set(&LLamaWrapper::CustomErrorLog, &this->sets);
    }

  
    ggml_backend_load_all();

    llama_numa_init(ggml_numa_strategy::GGML_NUMA_STRATEGY_DISABLED);
    llama_backend_init();
        	   
}

LLamaWrapper::~LLamaWrapper()
{
    this->Relase();
}

void LLamaWrapper::Relase()
{
    llama_batch_ext_free(batch);
    for (auto& sampler_config : sampler_configs) 
    {
        llama_sampler_free(sampler_config.sampler);
    }
    llama_free(ctx);
    llama_model_free(model);

    llama_backend_free();

    batch = nullptr;
    batchTokensCount = 0;

    sampler_configs.clear();
    ctx = nullptr;
    model = nullptr;
    vocab = nullptr;
}

void LLamaWrapper::ClearContext()
{
    if (ctx)
    {
        //llama_kv_self_clear(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);
    }
}


bool LLamaWrapper::LoadProgressCallback(float progress, void* user_data)
{
    return true;
}

/// <summary>
/// Print info for error codes for llama_process (see llama.h - llama_decode for error codes)
/// </summary>
/// <param name="errCode"></param>
/// <param name="fnSrc"></param>
void LLamaWrapper::LogDecodeProcessError(int32_t errCode, const char* fnSrc)
{
    MY_LOG_ERROR("%s : failed to eval, return code %d\n", fnSrc, errCode);
    if (errCode == 1)
    {
        MY_LOG_ERROR("could not find a KV slot for the batch (try reducing the size of the batch or increase the context)");
    }
    else if (errCode == 2)
    {
        MY_LOG_ERROR("aborted     (processed ubatches will remain in the context's memory)");
    }
    else if (errCode == -1)
    {
        MY_LOG_ERROR("invalid input batch");
    }
    else if (errCode < -1)
    {
        MY_LOG_ERROR("fatal error (processed ubatches will remain in the context's memory)");
    }
}

void LLamaWrapper::InitModel()
{
    if (this->LoadModel() == false)
    {
        throw std::runtime_error("load_model failed");
    }
}

/// <summary>
/// Load model
/// </summary>
/// <returns></returns>
bool LLamaWrapper::LoadModel()
{	
    this->Relase();

    //llama_model_params model_params = common_model_params_to_llama(params);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = &LLamaWrapper::LoadProgressCallback;
    model_params.n_gpu_layers = sets.gpuLayers;       
	model = llama_model_load_from_file(this->modelPath.c_str(), model_params);
    if (!model) 
    {
        MY_LOG_ERROR("Unable to load model %s", this->modelPath.c_str());
        return false;
    }

    vocab = llama_model_get_vocab(model);

    //int32_t n_ctx = 0; // context size, 0 == context the model was trained with
    //int32_t n_parallel = 1; //number of parallel sequences to decode (=> number of parallel batches)
    //int32_t n_predict = -1; // max. number of new tokens to predict, -1 == no limit (=> total length of the sequences including the prompt)
    //token_list = prompt tokens
    //const int n_kv_req = tokens_list.size() + (n_predict - tokens_list.size()) * n_parallel;
    

    /*
    Batch Size

    batch: Logical batch size.
    Increasing this value above the value of the physical batch size may improve prompt processing performance when
    using multiple GPUs with pipeline parallelism.
    Default : 2048.

    ubatch: Physical batch size.
    This is the maximum number of tokens that may be processed at a time.
    Increasing this value may improve performance during prompt processing, at the expense of higher memory usage.
    Default : 512.

    BEWARE: non-causal attention requires n_ubatch >= n_tokens
    */

    /*
    n_ctx(Context Size) : Defines the maximum number of tokens the model can "remember" at one time(prompt + generated output).
    n_batch(Batch Size) : Defines how many tokens are processed in parallel when ingesting the prompt.
    */

    // create an instance of llama_context    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = sets.ctxSize * sets.n_parallel;
    //ctx_params.n_batch = (sets.logicalBatchSize == 0) ? ctx_params.n_ctx : sets.logicalBatchSize;    
    ctx_params.n_batch = ctx_params.n_ctx;  //std::max<uint32_t>(sets.ctxSize, sets.n_parallel);    
    ctx_params.n_ubatch = (sets.physicalBatchSize == 0) ? ctx_params.n_ubatch : sets.physicalBatchSize;
    ctx_params.n_seq_max = sets.n_parallel;
    ctx_params.kv_unified = (sets.n_parallel > 1) ? true : false;

    ctx_params.no_perf = true;          // disable performance metrics    
    ctx_params.embeddings = sets.extractEmbeddings;

    if (sets.onContextSetup)
    {
        sets.onContextSetup(ctx_params);
    }

    ctx = llama_init_from_model(model, ctx_params);

    if (!ctx) 
    {
        MY_LOG_ERROR("Failed to create the llama_context");
        return false;
    }
   
    // initialize sampler
    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;      // disable performance metrics

    for (int32_t i = 0; i < sets.n_parallel; i++)
    {
        llama_sampler* sampler = llama_sampler_chain_init(sampler_params);
        
        if (sets.samplingTopP.has_value())
        {
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(*sets.samplingTopP, sets.samplingMinKeep));
        }

        if (sets.samplingMinP.has_value())
        {
            llama_sampler_chain_add(sampler, llama_sampler_init_min_p(*sets.samplingMinP, sets.samplingMinKeep));
        }

        if (sets.samplingTemperature.has_value())
        {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(*sets.samplingTemperature));
        }

        llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));


        sampler_configs.emplace_back(i, sampler);
    }

    /*
    if (params.sampling.backend_sampling) {
        ctx_params.samplers = sampler_configs.data();
        ctx_params.n_samplers = sampler_configs.size();
    }
    */

    const int n_ctx = this->GetContextSize();
    const int n_ctx_train = llama_model_n_ctx_train(model);

    if (n_ctx > n_ctx_train)
    {
        MY_LOG_WARNING("warning: model was trained on only % d context tokens(% d specified)",
            n_ctx_train, this->GetContextSize());
    }

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (ctx_params.n_ctx > n_ctx)
    {
        MY_LOG_ERROR("ctx_params.n_ctx (%d) > n_ctx (%d), the required KV cache size is not big enough", ctx_params.n_ctx, n_ctx);
        MY_LOG_ERROR("either reduce n_parallel or increase context size");
        return false;
    }

    // max batch size cannot exceed context size    
    GGML_ASSERT(ctx_params.n_batch >= ctx_params.n_ctx);

    this->ResetProcessedTokensCount();

    return true;
}

int LLamaWrapper::GetContextSize() const
{
    if (ctx == nullptr)
    {
        return 0;
    }
    return llama_n_ctx(ctx);    
}

int LLamaWrapper::GetVocabSize() const
{
    if (this->vocab == nullptr)
    {
        return 0;
    }

    return llama_vocab_n_tokens(this->vocab);
}

int LLamaWrapper::GetEmbeddingSize() const
{
    if (model == nullptr)
    {
        return 0;
    }
    
    return llama_model_n_embd(model);
}

/// <summary>
/// ctx size * parallel count
/// </summary>
/// <returns></returns>
int LLamaWrapper::GetMaxTokensForAllBatches() const
{
    if (model == nullptr)
    {
        return 0;
    }

    return llama_n_batch(ctx);
}

int LLamaWrapper::GetMaxParallelCount() const
{
    if (model == nullptr)
    {
        return 0;
    }

    return llama_n_seq_max(ctx);
}

//===========================================================================
// Tokenization
//===========================================================================

void LLamaWrapper::ResetProcessedTokensCount()
{
    this->processedInputTokensCount = 0;
    this->processedOutputTokensCount = 0;
}

size_t LLamaWrapper::GetProcessedInputTokensCount() const
{
    return this->processedInputTokensCount;
}

size_t LLamaWrapper::GetProcessedOutputTokensCount() const
{
    return this->processedOutputTokensCount;
}


llama_token LLamaWrapper::GetSepToken() const
{
    return llama_vocab_sep(vocab);
}

std::vector<llama_token> LLamaWrapper::Tokenize(std::string_view prompt)
{
    const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;
    //const bool is_first = llama_kv_self_used_cells(ctx) == 0;

    return this->Tokenize(prompt, is_first);
}

/// <summary>
/// Convert string to tokens
/// </summary>
/// <param name="prompt"></param>
/// <returns></returns>
std::vector<llama_token> LLamaWrapper::Tokenize(std::string_view prompt, bool addSpecial)
{

    // upper limit for the number of tokens
    int32_t n_tokens = static_cast<int32_t>(prompt.length() + 2 * addSpecial);
    
    std::vector<llama_token> result(n_tokens);

    n_tokens = llama_tokenize(vocab, 
        prompt.data(), static_cast<int32_t>(prompt.length()),
        result.data(), static_cast<int32_t>(result.size()),
        addSpecial, true);

    if (n_tokens < 0) 
    {
        result.resize(-n_tokens);
        int check = llama_tokenize(vocab, 
            prompt.data(), static_cast<int32_t>(prompt.length()),
            result.data(), static_cast<int32_t>(result.size()),
            addSpecial, true);
        GGML_ASSERT(check == -n_tokens);

        //processedInputTokensCount += n_tokens;
        processedInputTokensCount += result.size();
    }
    else 
    {
        processedInputTokensCount += n_tokens;
        result.resize(n_tokens);
    }

    return result;    
}

/// <summary>
/// Convert tokens to string
/// </summary>
/// <param name="new_token_id"></param>
/// <returns></returns>
std::string LLamaWrapper::TokenToString(llama_token new_token_id)
{
    // convert the token to a string, print it and add it to the response
    char buf[256] = { 0 };
    bool addSpecialTokensToOutput = false;

    int n = llama_token_to_piece(vocab,
        new_token_id, buf, sizeof(buf),
        0, addSpecialTokensToOutput);

    if (n < 0)
    {
        GGML_ABORT("failed to convert token to piece\n");
    }

    processedOutputTokensCount++;

    return std::string(buf, n);
}

//===========================================================================
// Batches
//===========================================================================

void LLamaWrapper::InitBatch(int32_t n_tokens)
{    
    const int n_batch = this->GetMaxTokensForAllBatches();
    if (n_tokens > n_batch)
    {
        MY_LOG_ERROR("Too many tokens to batch. Out of size");
        return;
    }

    if (batch != nullptr)
    {
        llama_batch_ext_free(batch);
    }

    //capacity comes from ctx n_batch
    batch = llama_batch_ext_init(this->ctx);    
}

std::vector<llama_seq_id> LLamaWrapper::InitSeqIds()
{
    std::vector<llama_seq_id> seq_ids(sets.n_parallel, 0);
    for (int32_t i = 0; i < sets.n_parallel; i++)
    {
        seq_ids[i] = i;
    }

    return seq_ids;
}

/// <summary>
/// Add data to batch
/// (copied from common_batch_add)
/// </summary>
/// <param name="batch"></param>
/// <param name="id"></param>
/// <param name="pos"></param>
/// <param name="seq_ids"></param>
/// <param name="logits"></param>
int32_t LLamaWrapper::AddToBatch(llama_token tokenId, llama_pos pos,
    const std::vector<llama_seq_id>& seq_ids, bool logits)
{    
    const int32_t idx = llama_batch_ext_add_token(batch, seq_ids[0], tokenId);
    llama_batch_ext_set_pos(batch, idx, &pos);

    for (size_t j = 1; j < seq_ids.size(); j++)
    {
        llama_batch_ext_add_seq(batch, idx, seq_ids[j]);
    }

    //const bool ok = embeddings
    // ? llama_batch_ext_set_output_embd(batch, idx, requested)
    //    : llama_batch_ext_set_output_logits(batch, idx, requested);

    llama_batch_ext_set_output_logits(batch, idx, logits);

    batchTokensCount++;

    return idx;
}

void LLamaWrapper::AddToBatch(const std::vector<llama_token>& tokens,
    const std::vector<llama_seq_id>& seq_ids, bool logits)
{
    int32_t n_tokens = static_cast<int32_t>(tokens.size());
   
    for (llama_pos i = 0; i < n_tokens - 1; i++)
    {
        this->AddToBatch(tokens[i], i, seq_ids, logits);
    }

    this->AddToBatch(tokens[n_tokens - 1], n_tokens - 1, seq_ids, true);    
}


/// <summary>
/// (copied from common_batch_clear)
/// </summary>
/// <param name="batch"></param>
void LLamaWrapper::ClearBatch()
{
    if (batch != nullptr)
    {
        llama_batch_ext_clear(batch);
    }

    batchTokensCount = 0;
}

#endif