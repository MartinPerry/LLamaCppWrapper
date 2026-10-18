#ifndef LLAMA_WRAPPER_H
#define LLAMA_WRAPPER_H

#if __has_include (<llama.h>)

#include <vector>
#include <list>
#include <string>
#include <optional>
#include <functional>

#include <llama.h>
#include <llama-cpp.h>


class LLamaWrapper 
{
public:

	struct Settings
	{				
		bool logEnabled = false;

		int32_t gpuLayers = 99; // number of layers to store in VRAM

		int32_t ctxSize = 2048;  // text context for single "parallel sequence", 0 = default from model
		
		int32_t physicalBatchSize = 0; //This is the maximum number of tokens that may be processed at a time, 0 = default value


		int32_t n_parallel = 1; //number of parallel sequences to decode

		bool extractEmbeddings = false;		

		int32_t samplingMinKeep = 1; // 0 = disabled, otherwise samplers should return at least min_keep tokens
		std::optional<float> samplingTopP = std::nullopt;// common default = 0.95f; | 1.0 = disabled
		std::optional<float> samplingMinP = 0.05f; //common default = 0.05f; | 0.0 = disabled
		std::optional<float> samplingTemperature = 0.8f;// common default = 0.80f; | <= 0.0 to sample greedily, 0.0 to not output probabilities
	
	
		std::function<void(llama_context_params&)> onContextSetup = nullptr; //can specify custom parts of llama_context_params before model is inited

	};

	static void CustomErrorLog(enum ggml_log_level level, const char* text, void* user_data);

	LLamaWrapper(const std::string& modelPath, const Settings& sets);
	virtual ~LLamaWrapper();

	virtual void Relase();
	virtual void ClearContext();
		
	virtual void InitModel();

	int GetContextSize() const;
	int GetVocabSize() const;
	virtual int GetEmbeddingSize() const;
	int GetMaxTokensForAllBatches() const;
	int GetMaxParallelCount() const;
	llama_token GetSepToken() const;
	
	void ResetProcessedTokensCount();
	size_t GetProcessedInputTokensCount() const;
	size_t GetProcessedOutputTokensCount() const;

protected:
	
	std::string modelPath;
	Settings sets;	

	// llama.cpp-specific types
	llama_context* ctx;
	llama_model* model;
	//llama_sampler* sampler;
	std::vector<llama_sampler_seq_config> sampler_configs;
	llama_batch_ext* batch;
	llama_token currToken;
	int32_t batchTokensCount;

	const llama_vocab* vocab;

	size_t processedInputTokensCount;
	size_t processedOutputTokensCount;

	static void LogDecodeProcessError(int32_t errCode, const char* fnSrc);
	static bool LoadProgressCallback(float progress, void* user_data);

	bool LoadModel();
	
	std::vector<llama_token> Tokenize(std::string_view prompt);
	std::vector<llama_token> Tokenize(std::string_view prompt, bool addSpecial);
	std::string TokenToString(llama_token new_token_id);
	

	void InitBatch(int32_t n_tokens);
		
	std::vector<llama_seq_id> InitSeqIds();

	virtual int32_t AddToBatch(llama_token tokensId, llama_pos pos,
		const std::vector<llama_seq_id>& seq_ids, bool logits);
	void AddToBatch(const std::vector<llama_token>& tokens,
		const std::vector<llama_seq_id>& seq_ids, bool logits);

	void ClearBatch();
};


#endif

#endif
