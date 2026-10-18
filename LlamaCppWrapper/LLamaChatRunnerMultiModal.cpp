#include "./LLamaChatRunnerMultiModal.h"

#if __has_include (<llama.h>) && __has_include (<mtmd.h>)

#include <stdexcept>

#include <mtmd-helper.h>

#include "./Utils/Logger.h"

#include "./LLamaChatMultiModal.h"

#pragma comment(lib, "mtmd.lib")

//Very simplified version
//not all models may work
//for example antiprompts are nt supported:
/*
// load antiprompt tokens for legacy templates
if (params.chat_template == "vicuna")
{
	antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
}
else if (params.chat_template == "deepseek")
{
	antiprompt_tokens = common_tokenize(lctx, "###", false, true);
}
*/


LLamaChatRunnerMultiModal::LLamaChatRunnerMultiModal(const std::string& modelPath, const std::string& projectorPath,
	const Settings& sets) : 
	LLamaChatRunner(modelPath, sets),
	projectorPath(projectorPath)	
{
		
}

LLamaChatRunnerMultiModal::~LLamaChatRunnerMultiModal()
{
}

void LLamaChatRunnerMultiModal::InitModel()
{
	LLamaChatRunner::InitModel();

	if (this->LoadImageModel() == false)
	{
		throw std::runtime_error("load_image_projector failed");
	}

	const int n_batch = this->GetMaxTokensForAllBatches();

	// initialize batch    
	this->InitBatch(n_batch);

	this->n_past_seq = std::vector<int>(sets.n_parallel, 0);
}

void LLamaChatRunnerMultiModal::InitChats()
{
	this->chats = std::vector<std::shared_ptr<LLamaChat>>(sets.n_parallel, nullptr);

	for (int i = 0; i < sets.n_parallel; i++)
	{
		chats[i] = std::shared_ptr<LLamaChatMultiModal>(new LLamaChatMultiModal(this));
	}

	this->ClearMessages();
}

bool LLamaChatRunnerMultiModal::LoadImageModel()
{			
	mtmd_context_params mparams = mtmd_context_params_default();
	mparams.use_gpu = sets.gpuLayers > 0;
	mparams.print_timings = sets.logEnabled;
	mparams.n_threads = 1;// params.cpuparams.n_threads;
	//mparams.verbosity = sets.logEnabled ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;
		
	mtmd_log_set(&LLamaWrapper::CustomErrorLog, &this->sets);

	auto mtmdModelCtx = mtmd_init_from_file(this->projectorPath.c_str(), model, mparams);

	ctx_vision.reset(mtmdModelCtx);
	
	if (!ctx_vision.get())
	{
		MY_LOG_ERROR("Failed to load vision model from %s", this->projectorPath.c_str());
		return false;
	}

	return true;
}


void LLamaChatRunnerMultiModal::EvalMtmd(llama_seq_id seqId, const std::string& prompt, mtmd::bitmaps& bmps)
{
	mtmd_input_text text;
	text.text = prompt.c_str();
	text.text_len = prompt.length();
	text.add_special = true;;
	text.parse_special = true;


	mtmd::input_chunks chunks(mtmd_input_chunks_init());
	auto bitmaps_c_ptr = bmps.c_ptr();

	int32_t res = mtmd_tokenize(ctx_vision.get(),
		chunks.ptr.get(), // output
		&text, // text
		bitmaps_c_ptr.data(),
		bitmaps_c_ptr.size());
	if (res != 0)
	{		
		MY_LOG_ERROR("Unable to tokenize prompt, res = %d", res);
		throw std::runtime_error("Unable to tokenize prompt");		
	}

	bmps.entries.clear();
	
	llama_pos new_n_past = 0;

	res = mtmd_helper_eval_chunks(ctx_vision.get(),
		ctx, // lctx
		chunks.ptr.get(), // chunks
		this->n_past_seq[seqId], // n_past
		seqId,
		this->GetMaxTokensForAllBatches(), // n_batch
		true, // logits_last
		&new_n_past);

	if (res != 0)
	{
		MY_LOG_ERROR("Unable to eval prompt");
		return;
	}

	this->n_past_seq[seqId] = new_n_past;
}

//=====================================================================================================
// Single processing
//=====================================================================================================

std::string LLamaChatRunnerMultiModal::Loop(const std::string& prompt)
{	
	auto chat = this->GetChatAs<LLamaChatMultiModal>(0);

	if (chat->bitmaps.entries.size() == 0)
	{
		return LLamaChatRunner::Loop(prompt);
	}

	this->EvalMtmd(0, prompt, chat->bitmaps);

	std::string response;

	//std::vector<llama_token> generated_tokens;
	while (true)
	{
		// sample the next token		
		llama_token token_id = llama_sampler_sample(sampler_configs[0].sampler, ctx, -1);

		//llama_token token_id = common_sampler_sample(ctx.sampler, ctx.lctx, -1);
		//generated_tokens.push_back(token_id);
		//common_sampler_accept(ctx.sampler, token_id, true);

		if (llama_vocab_is_eog(vocab, token_id))
		{
			//printf("\n");
			break; // end of generation
		}

		response += this->TokenToString(token_id);
		
		
		std::vector<llama_seq_id> seq_ids = this->InitSeqIds();

		// eval the token
		this->ClearBatch();
		this->AddToBatch(token_id, n_past_seq[0]++, seq_ids, true);
		
		auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
		if (r != 0)
		{			
			LLamaWrapper::LogDecodeProcessError(r, __func__);
			break;
		}		
	}

	return response;
}

//=====================================================================================================
// Parallel processing
//=====================================================================================================

/*
std::vector<std::string> LLamaChatRunnerMultiModal::Loop(const std::vector<std::string>& prompts)
{
	int imgCount = 0;
	for (size_t i = 0; i < this->chats.size(); i++)
	{
		auto chat = this->GetChatAs<LLamaChatMultiModal>(i);
		if (chat->bitmaps.entries.size() != 0)
		{
			imgCount++;
		}
	}

	if (imgCount == 0)
	{
		return LLamaChatRunner::Loop(prompts);
	}

	//mtmd not working with parallel
	
	std::vector<std::string> streams(sets.n_parallel, "");

	for (size_t i = 0; i < prompts.size(); i++)
	{
		streams[i] = this->Loop(prompts[i]);
	}
	
	return streams;
}
*/

std::vector<std::string> LLamaChatRunnerMultiModal::Loop(const std::vector<std::string>& prompts)
{
	int imgCount = 0;
	for (size_t i = 0; i < this->chats.size(); i++)
	{
		auto chat = this->GetChatAs<LLamaChatMultiModal>(static_cast<int>(i));
		if (chat->bitmaps.entries.size() != 0)
		{
			imgCount++;
		}
	}

	if (imgCount == 0)
	{
		return LLamaChatRunner::Loop(prompts);
	}

	//
	//------- ChatGpt ------
	//

	//todo: clenaup for methods that i already have

	std::vector<std::string> streams(sets.n_parallel, "");


	std::vector<int32_t>     i_batch(sets.n_parallel, -1);
	std::vector<uint8_t>     finished(sets.n_parallel, 0);

	// keep bitmaps for all prompts during tokenization/eval
	// (mtmd_tokenize does not take ownership; safe to reuse)
	const auto eval_text_seq = [&](int s, const std::string& prompt, llama_pos& n_past) -> void {
		auto toks = this->Tokenize(prompt, true);

		// eval token-by-token (simple, robust)
		for (size_t i = 0; i < toks.size(); ++i) 
		{
			const bool logits_last = (i + 1 == toks.size());
			this->ClearBatch();
			this->AddToBatch(toks[i], n_past++, { (llama_seq_id)s }, logits_last);			

			auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
			if (r != 0)
			{
				LLamaWrapper::LogDecodeProcessError(r, __func__);
				throw std::runtime_error("llama_decode failed");
			}			
		}
	};
	
	// === PREFILL (per-seq), and immediately sample first token while logits are still "current" ===
	std::vector<llama_token> first_tok(sets.n_parallel, -1);

	for (int s = 0; s < sets.n_parallel; ++s) 
	{

		auto chat = this->GetChatAs<LLamaChatMultiModal>(s);

		if (chat->bitmaps.entries.empty()) 
		{
			eval_text_seq(s, prompts[s], n_past_seq[s]);
		}
		else 
		{				
			this->EvalMtmd(s, prompts[s], chat->bitmaps);
		}


		// sample first generated token for this seq from the *current* logits
		const llama_token t0 = llama_sampler_sample(sampler_configs[s].sampler, ctx, -1);

		if (llama_vocab_is_eog(vocab, t0)) 
		{
			finished[s] = 1;
			first_tok[s] = -1;
		}
		else 
		{
			first_tok[s] = t0;
			streams[s] += this->TokenToString(t0);
		}
	}

	// if we have any active seqs, decode their first generated token in one batch to get per-seq logits
	{
		this->ClearBatch();
		
		for (int s = 0; s < sets.n_parallel; ++s) 
		{
			if (finished[s])
			{
				continue;
			}
			if (first_tok[s] == -1) 
			{ 
				finished[s] = 1; 
				continue; 
			}

			// add token at current position and request logits
			this->AddToBatch(first_tok[s], n_past_seq[s]++, { (llama_seq_id)s }, true);			

			i_batch[s] = batchTokensCount - 1;			
		}

		if (batchTokensCount > 0)
		{
			auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
			if (r != 0)
			{
				LLamaWrapper::LogDecodeProcessError(r, __func__);
				return {};
			}
		}
	}

	// === MAIN PARALLEL GENERATION LOOP ===
	while (true) 
	{
		this->ClearBatch();
		
		for (int s = 0; s < sets.n_parallel; ++s) 
		{
			if (finished[s])
			{
				continue;
			}

			const llama_token tok = llama_sampler_sample(sampler_configs[s].sampler, ctx, i_batch[s]);

			if (llama_vocab_is_eog(vocab, tok)) 
			{
				finished[s] = 1;
				continue;
			}

			streams[s] += this->TokenToString(tok);

			this->AddToBatch(tok, n_past_seq[s]++, { (llama_seq_id)s }, true);
			
			i_batch[s] = batchTokensCount - 1;			
		}

		if (batchTokensCount == 0)
		{
			break;
		}

		auto r = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
		if (r != 0)
		{
			MY_LOG_ERROR("failed to decode parallel batch");
			LLamaWrapper::LogDecodeProcessError(r, __func__);
			break;
		}		
	}

	std::vector<std::string> response(prompts.size());
	for (int32_t i = 0; i < sets.n_parallel; ++i)
	{
		response[i] += streams[i];
	}

	return response;
}

#endif