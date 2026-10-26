#include "./LLamaDecision.h"

#if __has_include (<llama.h>)

//https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp

// Adapted from the supplied server-decision.cpp/.h and the decision output
// extraction in server-context.cpp. No server, HTTP, task queue or mtmd linkage.
#include <llama-ext.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <utility>


#include "../3rdParty/json.h"
#include "../3rdParty/jinja/jparser.h"

#include "../Utils/Logger.h"

#include "./LlamaDecisionContext.h"
#include "./LLamaDecisionContextMultiModal.h"

//============================================================================================================================
//============================================================================================================================
//============================================================================================================================


LLamaWrapper::Settings LLamaDecision::PrepareSettings(const Settings& settings)
{
	// Questions are evaluated sequentially. No sampling or conversation slots.

	Settings result = settings;	
	result.n_parallel = 1;
	result.extractEmbeddings = false;
	result.samplingTopP.reset();
	result.samplingMinP.reset();
	result.samplingTemperature.reset();
	
	const auto setup = settings.onContextSetup;
	result.onContextSetup = [setup](llama_context_params& ctx_params) {
		
		if (setup) 
		{
			setup(ctx_params);
		}
		// Raw token/head outputs are required; pooling and backend sampling
		// would change what the decision scorer reads.
		ctx_params.pooling_type = LLAMA_POOLING_TYPE_NONE;
		ctx_params.samplers = nullptr;
		ctx_params.n_samplers = 0;
	};
	return result;
}

LLamaDecision::LLamaDecision(const std::string& modelPath, const Settings& settings) :
	LLamaWrapper(modelPath, PrepareSettings(settings))
{
	InitModel();
	
}

LLamaDecision::~LLamaDecision()
{
}

void LLamaDecision::InitModel()
{
	LLamaWrapper::InitModel();
				
	if (llama_model_has_encoder(model) && llama_model_has_decoder(model))
	{
		throw std::invalid_argument("encoder-decoder decision models are not supported");
	}		
	
	dcx = DecisionContext::Create(model);	
}


void LLamaDecision::SetCustomPromptBuilder(CustomPromptBuilder callback)
{	
	dcx->customPromptBuilder = std::move(callback);
}

std::unordered_map<std::string, DecisionAnswer> LLamaDecision::Run(const std::string& state, 
	const std::vector<Question>& inputQuestions)
{
    return RunInternal(state, inputQuestions);
}

std::unordered_map<std::string, DecisionAnswer> LLamaDecision::Run(const std::vector<Question>& inputQuestions)
{
    return RunInternal("", inputQuestions);
}

std::unordered_map<std::string, DecisionAnswer> LLamaDecision::RunInternal(const std::string& state,
    const std::vector<Question>& inputQuestions)
{        
	if (dcx->SetNullState((state == "")) == false)
	{
		return {};
	}

	const auto questions = dcx->PrepareQuestions(inputQuestions);
	if (questions.empty())
	{
		return {};
	}
	
	// Prepare and validate every prompt before evaluating any question.
	std::vector<DecisionPrompt> prompts;
	if (dcx->IsJointTask()) 
	{
		prompts.emplace_back();
		dcx->FillTask(state, questions, questions[0], 0, prompts.back());
	}
	else 
	{
		for (const auto& question : questions) 
		{
			auto vn = dcx->GetVariantsCount(question);
			for (size_t variant = 0; variant < vn; variant++)
			{
				prompts.emplace_back();
				dcx->FillTask(state, questions, question, variant, prompts.back());
			}
		}
	}


    for (const auto& prompt : prompts) 
	{
        if (prompt.tokens.empty() || (prompt.labels.empty() && prompt.markers.empty() && prompt.order.empty())) 
		{
            MY_LOG_ERROR("decision prompt has no model outputs");
            return {};
        }
    }

	try 
	{
		std::vector<std::vector<float>> outputs;
		for (const auto& prompt : prompts)
		{
			auto res = this->Evaluate(ctx, model, prompt);
            if (res.empty()) {
                if (auto memory = llama_get_memory(ctx)) llama_memory_clear(memory, true);
                return {};
            }
			outputs.push_back(std::move(res));

			processedInputTokensCount += prompt.tokens.size();
		}
		std::unordered_map<std::string, DecisionAnswer> answers;

		size_t resultIndex = 0;
		size_t scoreIndex = 0;
		for (const auto& question : questions)
		{
			std::vector<std::vector<float>> scores;
			if (dcx->IsJointTask())
			{
				const auto& joint = outputs.at(0);
				if (scoreIndex + question.criteria.size() > joint.size())
				{
					throw std::runtime_error("joint decision returned too few scores");
				}
				scores.emplace_back(joint.begin() + scoreIndex, joint.begin() + scoreIndex + question.criteria.size());
				scoreIndex += question.criteria.size();
			}
			else
			{
				auto vn = dcx->GetVariantsCount(question);
				for (size_t variant = 0; variant < vn; variant++)
				{
					scores.push_back(std::move(outputs.at(resultIndex++)));
				}
			}
			answers.try_emplace(question.id, dcx->FormatAnswer(question, scores));
		}

		if (dcx->IsJointTask() && scoreIndex != outputs.at(0).size())
		{
			throw std::runtime_error("joint decision returned too many scores");
		}
		if (auto memory = llama_get_memory(ctx))
		{
			llama_memory_clear(memory, true);
		}
		return answers;
	}
	catch (...) 
	{
		if (auto memory = llama_get_memory(ctx)) 
		{
			llama_memory_clear(memory, true);
		}
		throw;
	}
}

std::vector<float> LLamaDecision::Evaluate(llama_context* ctx, const llama_model* model,
	const DecisionPrompt& inputPrompt)
{
    DecisionPrompt prompt = inputPrompt;
	if (auto memory = llama_get_memory(ctx))
	{
		llama_memory_clear(memory, true);
	}
	
	const bool embeddings = prompt.labels.empty();
	llama_set_embeddings(ctx, embeddings);

    if (!this->ProcessPrompt(prompt)) return {};
    return this->ReadScores(ctx, model, prompt);
}

bool LLamaDecision::ProcessPrompt(DecisionPrompt& prompt)
{
    const bool embeddings = prompt.labels.empty();
	this->InitBatch(this->GetMaxTokensForAllBatches());

	
	for (size_t i = 0; i < prompt.tokens.size(); i++) 
	{
		const bool requested = embeddings || i + 1 == prompt.tokens.size();

		const int32_t index = this->AddToBatch(prompt.tokens[i], i, {0}, requested);
		
		if (!prompt.order.empty() && !llama_batch_ext_set_decision_order(batch, index,
			static_cast<llama_decision_order>(prompt.order[i])))
		{
			throw std::runtime_error("failed to set joint decision order");
		}
	}

	const auto processType = llama_model_has_encoder(model)
		? LLAMA_PROCESS_TYPE_ENCODE : LLAMA_PROCESS_TYPE_DECODE;

	const int32_t status = llama_process(ctx, processType, batch);
	if (status != 0) 
	{
		throw std::runtime_error("decision evaluation failed (status " + std::to_string(status) + ")");
	}
    return true;
}

std::vector<float> LLamaDecision::ReadScores(llama_context* ctx, const llama_model* model,
    const DecisionPrompt& prompt)
{
    const bool embeddings = prompt.labels.empty();
	std::vector<float> scores;
	if (!embeddings) 
	{
		const float* logits = llama_get_logits_ith(ctx, static_cast<int32_t>(prompt.tokens.size() - 1));
		if (!logits) 
		{
			throw std::runtime_error("failed to read decision logits");
		}
		const int32_t vocabSize = llama_vocab_n_tokens(llama_model_get_vocab(model));
		for (const auto label : prompt.labels)
		{
			if (label < 0 || label >= vocabSize) 
			{
				throw std::runtime_error("invalid decision label token");
			}
			scores.push_back(logits[label]);
		}
		return DecisionContext::AggregateLabelScores(scores, prompt.label_groups);
	}

	auto embeddingAt = [&](int32_t position) -> const float* {
		if (position < 0 || static_cast<size_t>(position) >= prompt.tokens.size()) {
			throw std::runtime_error("decision output position is out of range");
		}
		const float* data = llama_get_embeddings_ith(ctx, position);
		if (!data) {
			throw std::runtime_error("failed to read decision embeddings");
		}
		return data;
	};

	if (!prompt.order.empty())
	{
		for (int32_t i = 0; i < prompt.n_scores; ++i)
		{
			scores.push_back(embeddingAt(i)[0]);
		}
		return scores;
	}

	const int32_t dimension = llama_model_n_embd_out(model);
	if (prompt.column < 0 || prompt.column >= dimension)
	{
		throw std::runtime_error("invalid decision embedding column");
	}

	const int32_t pointerSize = dimension / 2;
	const float* question = prompt.pointer >= 0 ? embeddingAt(prompt.pointer) : nullptr;
	if (question && (pointerSize == 0 || dimension % 2 != 0)) 
	{
		throw std::runtime_error("invalid decision pointer embedding dimension");
	}

	for (const int32_t marker : prompt.markers)
	{
		const float* option = embeddingAt(marker);
		if (!question) 
		{
			scores.push_back(option[prompt.column]);
		}
		else 
		{
			float dot = 0.0f;
			for (int32_t i = 0; i < pointerSize; ++i) 
			{
				dot += question[i] * option[pointerSize + i];
			}
			scores.push_back(dot / std::sqrt(static_cast<float>(pointerSize)));
		}
	}
	return scores;
}

#endif
