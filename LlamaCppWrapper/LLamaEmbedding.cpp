#include "./LLamaEmbedding.h"

#if __has_include (<llama.h>)

#include <cmath>

#include "./Utils/Logger.h"

float LLamaEmbedding::CosineSimilarity(const Embedding& embd1, const Embedding& embd2, int index)
{
    if (embd1.embedDim != embd2.embedDim)
    {
        return std::numeric_limits<float>::max();
    }

    const float* d1 = embd1.rawData.data() + embd1.startOffsets[index];
    const float* d2 = embd2.rawData.data() + embd2.startOffsets[index];

    return LLamaEmbedding::CosineSimilarity(d1, d2, embd1.embedDim);
}

float LLamaEmbedding::CosineSimilarity(const float* embd1, const float* embd2, int n)
{
    double sum = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;

    for (int i = 0; i < n; i++)
    {
        sum += embd1[i] * embd2[i];
        sum1 += embd1[i] * embd1[i];
        sum2 += embd2[i] * embd2[i];
    }

    // Handle the case where one or both vectors are zero vectors
    if (sum1 == 0.0 || sum2 == 0.0)
    {
        if (sum1 == 0.0 && sum2 == 0.0)
        {
            return 1.0f; // two zero vectors are similar
        }
        return 0.0f;
    }

    return static_cast<float>(sum / (std::sqrt(sum1) * std::sqrt(sum2)));
}

LLamaEmbedding::LLamaEmbedding(const std::string& modelPath, const Settings& sets) :
	LLamaWrapper(modelPath, sets),
    cutOfWhenExceedBatch(false),
    normalization(Normalization::EUCLIDEAN),
    pNorm(2),
    matryoshkaSize(-1)
{
    if (sets.extractEmbeddings == false)
    {
        MY_LOG_ERROR("extractEmbeddings must be enabled for emedding model");
        abort();
    }

    this->InitModel();

    if (llama_model_has_encoder(model) && llama_model_has_decoder(model))
    {
        MY_LOG_ERROR("Computing embeddings in encoder-decoder models is not supported");       
    }    
}

LLamaEmbedding::~LLamaEmbedding()
{
}

void LLamaEmbedding::SetNormalizationMode(Normalization n)
{
    this->normalization = n;
}

void LLamaEmbedding::SetNormalizationMode(Normalization n, int pNorm)
{
    this->normalization = n;
    this->pNorm = pNorm;
}

void LLamaEmbedding::EnableCutOffWhenExceedBatch(bool val)
{
    this->cutOfWhenExceedBatch = val;
}

int LLamaEmbedding::GetMaxParallelCount() const
{
    return LLamaWrapper::GetMaxParallelCount();
}

size_t LLamaEmbedding::GetProcessedTokensCount() const
{
    return LLamaWrapper::GetProcessedInputTokensCount() + LLamaWrapper::GetProcessedOutputTokensCount();
}

/// <summary>
/// For  models that supports Matryoshka embeding sizes, 
/// you can set embed dimension
/// https://huggingface.co/blog/matryoshka
/// </summary>
/// <param name="s"></param>
void LLamaEmbedding::SetMatryoshkaSize(int s)
{
    this->matryoshkaSize = s;
}

/// <summary>
/// Get embedding size 
/// If matryoshka is set, return matryoshka size
/// </summary>
/// <returns></returns>
int LLamaEmbedding::GetEmbeddingSize() const
{
    if (this->matryoshkaSize > 0)
    {
        return this->matryoshkaSize;
    }

    return LLamaWrapper::GetEmbeddingSize();
}

/// <summary>
/// Pooling explained
/// https://blog.ml6.eu/the-art-of-pooling-embeddings-c56575114cf8
/// </summary>
/// <returns></returns>
enum llama_pooling_type LLamaEmbedding::GetPoolingType() const
{
    return llama_pooling_type(ctx);
}


void LLamaEmbedding::Normalize(const float* inp, float* out, int n) const
{
    double sum = 0.0;

    switch (normalization) 
    {
        case Normalization::NONE:
            sum = 1.0;
            break;
        case Normalization::MAX_ABSOLUTE:
            for (int i = 0; i < n; i++) 
            {
                if (sum < std::abs(inp[i])) 
                {
                    sum = std::abs(inp[i]);
                }
            }
            sum /= 32760.0; // make an int16 range
            break;
        case Normalization::EUCLIDEAN:
            for (int i = 0; i < n; i++) 
            {
                sum += inp[i] * inp[i];
            }
            sum = std::sqrt(sum);
            break;
        default: // p-norm (euclidean is p-norm p=2)
            for (int i = 0; i < n; i++) 
            {
                sum += std::pow(std::abs(inp[i]), pNorm);
            }
            sum = std::pow(sum, 1.0 / pNorm);
            break;
    }

    if (sets.logEnabled)
    {
        printf("%f\n", sum);
    }

    const float norm = (sum > 0.0) ? static_cast<float>(1.0 / sum) : 0.0f;

    for (int i = 0; i < n; i++) 
    {
        out[i] = inp[i] * norm;
    }
}


LLamaEmbedding::Embedding LLamaEmbedding::Run(const std::string& data)
{
    return this->Run(std::vector<std::string_view>{ data });
}

LLamaEmbedding::Embedding LLamaEmbedding::Run(std::string_view data)
{
    return this->Run(std::vector<std::string_view>{ data });
}

template <typename T>
LLamaEmbedding::Embedding LLamaEmbedding::Run(const std::vector<T>& data)
{
    if (data.size() > sets.n_parallel)
    {        
        const int n_embd = this->GetEmbeddingSize();

        LLamaEmbedding::Embedding res;
        res.embedDim = n_embd;
        res.rawData.reserve(n_embd * data.size());

        for (size_t i = 0; i < data.size(); i += sets.n_parallel)
        {
            auto last = std::min(data.size(), i + sets.n_parallel);
            auto b = std::vector<std::string_view>(data.begin() + i, data.begin() + last);

            LLamaEmbedding::Embedding resBatch = this->Run(b);
           
            res.rawData.insert(res.rawData.end(), resBatch.rawData.begin(), resBatch.rawData.end());

            auto offset = (res.startOffsets.empty()) ? 0 : res.startOffsets.back() + res.embedDim;
            for (size_t j = 0; j < resBatch.startOffsets.size(); j++)
            {
                res.startOffsets.push_back(offset + resBatch.startOffsets[j]);
            }            
        }
        
        return res;
    }
    else
    {
        std::vector<std::vector<llama_token>> batchTokens = this->BuildBatchTokens(data);

        LLamaEmbedding::Embedding res = this->BuildEmbedding(batchTokens, data.size(), data);
        return res;
    }    
}

void LLamaEmbedding::DecodeBatch(float* output, int n_seq, int n_embd)
{
    const enum llama_pooling_type pooling_type = this->GetPoolingType();

    // clear previous kv_cache values (irrelevant for embeddings)    
    //llama_kv_self_clear(ctx);
    
    // run model
    //LOG_INF("%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) 
    {
        // encoder-only model
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_ENCODE, batch) < 0)
        {
            MY_LOG_ERROR("failed to encode");
        }
    }
    else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) 
    {
        auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
        if (r != 0)
        {
            LLamaWrapper::LogDecodeProcessError(r, __func__);
        }       
    }

    const float* embd = nullptr;
    int embd_pos = -1;
    
    
    if (pooling_type == LLAMA_POOLING_TYPE_NONE)
    {
        for (int i = 0; i < batchTokensCount; i++)
        {
            const float* embd = llama_get_embeddings_ith(ctx, i);
            GGML_ASSERT(embd != nullptr);

            float* out = output + i * n_embd;
            this->Normalize(embd, out, n_embd);            
        }
    }
    else
    {
        for (int seq = 0; seq < n_seq; seq++)
        {
            const float* embd = llama_get_embeddings_seq(ctx, seq);

            if (!embd)
            {
                MY_LOG_ERROR("Missing embedding for sequence %d, batch tokens %d, sequences %d",
                    seq, batchTokensCount, n_seq);

                continue;
            }

            float* out = output + seq * n_embd;
            this->Normalize(embd, out, n_embd);            
        }
    }
    
    /*
    for (int i = 0; i < batchTokensCount; i++)
    {
        //if (!batch.logits[i]) 
        //{
        //    continue;
        //}
        
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) 
        {
            // try to get token embeddings
            embd_pos = i;
            embd = llama_get_embeddings_ith(ctx, embd_pos);
            GGML_ASSERT(embd != nullptr && "failed to get token embeddings");
        }
        else 
        {            
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            if (firstSeqId[i] == embd_pos)
            {
                //already obtained in the last iteration
                continue;
            }
            embd_pos = firstSeqId[i];
            embd = llama_get_embeddings_seq(ctx, embd_pos);
            
            GGML_ASSERT(embd != nullptr && "failed to get sequence embeddings - supported only when pooling_type is not NONE");
        }

        float* out = output + embd_pos * n_embd;
        this->Normalize(embd, out, n_embd);
    }
    */
}

int32_t LLamaEmbedding::AddToBatch(llama_token tokensId, llama_pos pos,
    const std::vector<llama_seq_id>& seq_ids, bool logits)
{
    if (firstSeqId.size() < (batchTokensCount + 1))
    {
        firstSeqId.resize(batchTokensCount + 1);
    }
    firstSeqId[batchTokensCount] = seq_ids[0];

    return LLamaWrapper::AddToBatch(tokensId, pos, seq_ids, logits);
}

//========================================================================================

template <typename T>
std::vector<std::vector<llama_token>> LLamaEmbedding::BuildBatchTokens(const std::vector<T>& data)
{
    const uint64_t n_batch = this->GetMaxTokensForAllBatches();

    std::vector<std::vector<llama_token>> batchTokens;

    for (const auto& q : data)
    {
        std::vector<llama_token> tokens = this->GetBatchTokens(q, n_batch);

        batchTokens.emplace_back(std::move(tokens));
    }

    return batchTokens;
}


std::vector<llama_token> LLamaEmbedding::GetBatchTokens(std::string_view text, const uint64_t n_batch)
{
    std::vector<llama_token> tokens = this->Tokenize(text, true);

    if (tokens.size() > n_batch)
    {
        MY_LOG_WARNING("Number of tokens in input line (%lld) exceeds batch size (%lld)\n",
            (long long int) tokens.size(), (long long int) n_batch);
        if (cutOfWhenExceedBatch == false)
        {
            MY_LOG_ERROR("Increase batch size and re-run");
            return {};
        }
        else 
        {
            MY_LOG_WARNING("Will auto truncate to batch_size");
        }

        tokens.resize(n_batch);
    }

    // check if the last token is SEP
    // it should be automatically added by the tokenizer when 'tokenizer.ggml.add_eos_token' is set to 'true'
    if (tokens.empty() || tokens.back() != this->GetSepToken())
    {
        //MY_LOG_WARNING("last token in the prompt is not SEP");
        //MY_LOG_WARNING("'tokenizer.ggml.add_eos_token' should be set to 'true' in the GGUF header");                
    }

    return tokens;
}

template <typename T>
LLamaEmbedding::Embedding LLamaEmbedding::BuildEmbedding(
    const std::vector<std::vector<llama_token>>& batchTokens, size_t n_prompts,
    const std::vector<T>& data)
{
    if (n_prompts > sets.n_parallel)
    {
        return {};
    }
    
    const enum llama_pooling_type pooling_type = this->GetPoolingType();
    
    // count number of embeddings
    size_t n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE)
    {
        for (int k = 0; k < n_prompts; k++)
        {
            n_embd_count += batchTokens[k].size();
        }
    }
    else
    {
        n_embd_count = n_prompts;
    }




    // allocate output
    const int n_embd = this->GetEmbeddingSize();
    
    Embedding res;
    res.embedDim = n_embd;
    res.rawData = std::vector<float>(n_embd_count * n_embd, 0);
    for (size_t i = 0; i < n_embd_count; i++)
    {
        res.startOffsets.push_back(static_cast<int>(i * n_embd));
    }

    float* emb = res.rawData.data();

    // break into batches

    const int n_batch = this->GetMaxTokensForAllBatches();
    
    // initialize batch    
    this->ClearContext();
    this->InitBatch(n_batch);

    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++)
    {
        // clamp to n_batch tokens
        auto& inp = batchTokens[k];

        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batchTokensCount + n_toks > n_batch)
        {
            float* out = emb + e * n_embd;
            this->DecodeBatch(out, s, n_embd);
            e += (pooling_type == LLAMA_POOLING_TYPE_NONE) ? batchTokensCount : s;
            s = 0;

            this->ClearContext();
            this->ClearBatch();
        }

        // add to batch - logits are added as true only for the last token
        LLamaWrapper::AddToBatch(inp, { s }, false);
        
        s += 1;
    }

    // final batch
    float* out = emb + e * n_embd;
    this->DecodeBatch(out, s, n_embd);

    if (sets.logEnabled)
    {
        MY_LOG("\n");

        if (pooling_type == LLAMA_POOLING_TYPE_NONE)
        {
            for (int j = 0; j < n_embd_count; j++)
            {
                MY_LOG("embedding %d: ", j);
                for (int i = 0; i < std::min(3, n_embd); i++)
                {
                    if (normalization == Normalization::NONE)
                    {
                        MY_LOG("%6.0f ", emb[j * n_embd + i]);
                    }
                    else {
                        MY_LOG("%9.6f ", emb[j * n_embd + i]);
                    }
                }
                MY_LOG(" ... ");
                for (int i = n_embd - 3; i < n_embd; i++)
                {
                    if (normalization == Normalization::NONE)
                    {
                        MY_LOG("%6.0f ", emb[j * n_embd + i]);
                    }
                    else {
                        MY_LOG("%9.6f ", emb[j * n_embd + i]);
                    }
                }
                MY_LOG("\n");
            }
        }
        else if (pooling_type == LLAMA_POOLING_TYPE_RANK)
        {
            for (int j = 0; j < n_embd_count; j++)
            {
                // NOTE: if you change this log - update the tests in ci/run.sh
                MY_LOG("rerank score %d: %8.3f\n", j, emb[j * n_embd]);
            }
        }
        else {

            // print the first part of the embeddings or for a single prompt, the full embedding
            for (int j = 0; j < n_prompts; j++)
            {
                MY_LOG("embedding %d: ", j);
                for (int i = 0; i < (n_prompts > 1 ? std::min(16, n_embd) : n_embd); i++)
                {
                    if (normalization == Normalization::NONE)
                    {
                        MY_LOG("%6.0f ", emb[j * n_embd + i]);
                    }
                    else
                    {
                        MY_LOG("%9.6f ", emb[j * n_embd + i]);
                    }
                }
                MY_LOG("\n");
            }


            // print cosine similarity matrix
            if (n_prompts > 1) 
            {
                MY_LOG("\n");
                MY_LOG("cosine similarity matrix:\n\n");
                for (int i = 0; i < n_prompts; i++)
                {
                    MY_LOG("%6.6s ", &data[i][0]);
                }
                MY_LOG("\n");
                for (int i = 0; i < n_prompts; i++)
                {
                    for (int j = 0; j < n_prompts; j++)
                    {
                        float sim = this->CosineSimilarity(emb + i * n_embd, emb + j * n_embd, n_embd);
                        MY_LOG("%6.2f ", sim);
                    }
                    MY_LOG("%1.10s", &data[i][0]);
                    MY_LOG("\n");
                }
            }
        }
    }

    return res;
}




template std::vector<std::vector<llama_token>> LLamaEmbedding::BuildBatchTokens(const std::vector<std::string>& data);
template std::vector<std::vector<llama_token>> LLamaEmbedding::BuildBatchTokens(const std::vector<std::string_view>& data);


template LLamaEmbedding::Embedding LLamaEmbedding::Run(const std::vector<std::string>& data);
template LLamaEmbedding::Embedding LLamaEmbedding::Run(const std::vector<std::string_view>& data);


template LLamaEmbedding::Embedding LLamaEmbedding::BuildEmbedding(
    const std::vector<std::vector<llama_token>>& batchTokens, size_t n_prompts,
    const std::vector<std::string>& data);
template LLamaEmbedding::Embedding LLamaEmbedding::BuildEmbedding(
    const std::vector<std::vector<llama_token>>& batchTokens, size_t n_prompts,
    const std::vector<std::string_view>& data);

#endif