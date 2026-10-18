#ifndef LLAMA_EMBEDDING_H
#define LLAMA_EMBEDDING_H

#if __has_include (<llama.h>)

#include <vector>
#include <list>
#include <string>

#include <llama.h>
#include <llama-cpp.h>

#include "./LLamaWrapper.h"

class LLamaEmbedding : protected LLamaWrapper
{
public:
	
	enum class Normalization 
	{
		NONE,
		MAX_ABSOLUTE,
		EUCLIDEAN,
		P_NORM
	};

	struct Embedding
	{
		int embedDim;
		std::vector<float> rawData;
		std::vector<int> startOffsets;
	};

	static float CosineSimilarity(const Embedding& embd1, const Embedding& embd2, int index = 0);
	static float CosineSimilarity(const float* embd1, const float* embd2, int n);

	LLamaEmbedding(const std::string& modelPath, const Settings& sets);
	~LLamaEmbedding();

	void SetNormalizationMode(Normalization n);
	void SetNormalizationMode(Normalization n, int pNorm);
	void SetMatryoshkaSize(int s);
	void EnableCutOffWhenExceedBatch(bool val);
	
	int GetMaxParallelCount() const;	
	int GetEmbeddingSize() const override;
	size_t GetProcessedTokensCount() const;

	Embedding Run(const std::string& data);
	Embedding Run(std::string_view data);

	template <typename T>	
	Embedding Run(const std::vector<T>& data);

protected:
	bool cutOfWhenExceedBatch;

	Normalization normalization;
	int pNorm;
	int matryoshkaSize;
	
	std::vector<llama_seq_id> firstSeqId;

	enum llama_pooling_type GetPoolingType() const;

	void Normalize(const float* inp, float* out, int n) const;
	void DecodeBatch(float* output, int n_seq, int n_embd);

	template <typename T>
	std::vector<std::vector<llama_token>> BuildBatchTokens(const std::vector<T>& data);	
	std::vector<llama_token> GetBatchTokens(std::string_view text, const uint64_t n_batch);

	template <typename T>
	Embedding BuildEmbedding(const std::vector<std::vector<llama_token>>& batchTokens, size_t n_prompts,
		const std::vector<T>& data);

	int32_t AddToBatch(llama_token tokensId, llama_pos pos,
		const std::vector<llama_seq_id>& seq_ids, bool logits) override;
};


#endif

#endif
