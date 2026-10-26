#include "./LlamaDecisionContext.h"

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

#include "./LLamaDecisionTemplate.h"

#include "./Models/LLamaDecisionGraniteSwitch.h"
#include "./Models/LLamaDecisionPplxDecider.h"
#include "./Models/LLamaDecisionVllmDecision2.h"
#include "./Models/LLamaDecisionVllmD3.h"
#include "./Models/LLamaDecisionLfm2D1.h"
#include "./Models/LLamaDecisionLfm2D1Omni.h"
#include "./Models/LLamaDecisionLaya.h"
#include "./Models/LLamaDecisionLev.h"
#include "./Models/LLamaDecisionNimble.h"
#include "./Models/LLamaDecisionClef.h"
#include "./Models/LLamaDecisionKev.h"
#include "./Models/LLamaDecisionOpenJev.h"


#include "../Utils/Logger.h"



bool DecisionContext::starts_with(const std::string& value, const std::string& prefix)
{
	return value.compare(0, prefix.size(), prefix) == 0;
}

void DecisionContext::replace_all(std::string& value, const std::string& search, const std::string& replacement) {
	if (search.empty()) return;
	size_t position = 0;
	while ((position = value.find(search, position)) != std::string::npos) {
		value.replace(position, search.size(), replacement);
		position += replacement.size();
	}
}

std::vector<std::string> DecisionContext::split(const std::string& value, const std::string& separator) {
	if (separator.empty()) return { value };
	std::vector<std::string> pieces;
	size_t begin = 0;
	size_t end;
	while ((end = value.find(separator, begin)) != std::string::npos) {
		pieces.push_back(value.substr(begin, end - begin));
		begin = end + separator.size();
	}
	pieces.push_back(value.substr(begin));
	return pieces;
}

std::vector<llama_token> DecisionContext::tokenize(const llama_vocab* vocab, const std::string& text,
	bool addSpecial, bool parseSpecial) 
{
	if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) 
	{
		throw std::invalid_argument("decision text is too long to tokenize");
	}

	const int32_t length = static_cast<int32_t>(text.size());
	const int32_t needed = llama_tokenize(vocab, text.data(), length, nullptr, 0, addSpecial, parseSpecial);

	if (needed == 0)
	{
		return {};
	}
	if (needed > 0 || needed == std::numeric_limits<int32_t>::min()) 
	{
		throw std::runtime_error("failed to measure decision tokens");
	}

	std::vector<llama_token> tokens(static_cast<size_t>(-needed));
	const int32_t count = llama_tokenize(vocab, text.data(), length, tokens.data(), -needed, addSpecial, parseSpecial);
	if (count < 0 || count > -needed) 
	{
		throw std::runtime_error("failed to tokenize decision text");
	}
	tokens.resize(count);
	return tokens;
}

std::string DecisionContext::token_to_piece(const llama_vocab* vocab, llama_token token) {
	std::vector<char> buffer(128);
	int32_t count = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
	if (count == std::numeric_limits<int32_t>::min()) {
		throw std::runtime_error("invalid decision marker token");
	}
	if (count < 0) {
		buffer.resize(static_cast<size_t>(-count));
		count = llama_token_to_piece(vocab, token, buffer.data(), static_cast<int32_t>(buffer.size()), 0, true);
	}
	if (count < 0 || static_cast<size_t>(count) > buffer.size()) {
		throw std::runtime_error("failed to read decision marker token");
	}
	return std::string(buffer.data(), count);
}




std::string DecisionContext::meta_str(const llama_model* model, const std::string& key) {
	char buffer[256];
	const int32_t n = llama_model_meta_val_str(model, key.c_str(), buffer, sizeof(buffer));
	if (n < 0) {
		return "";
	}
	if (static_cast<size_t>(n) < sizeof(buffer)) {
		return std::string(buffer, n);
	}
	std::vector<char> large(static_cast<size_t>(n) + 1);
	const int32_t copied = llama_model_meta_val_str(model, key.c_str(), large.data(), large.size());
	if (copied != n) {
		throw std::runtime_error("failed to read decision metadata: " + key);
	}
	return std::string(large.data(), n);
}




std::string DecisionContext::escape_special_tokens(const std::string & text) {
    const auto is_name = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
    };
    std::string out;
    size_t n_done = 0; // the chars of text that are in out
    for (size_t i = text.find("<|"); i != std::string::npos; i = text.find("<|", std::max(i + 1, n_done))) {
        size_t j = i + 2;
        while (j < text.size() && is_name(text[j])) {
            j++;
        }
        if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
            out.append(text, n_done, i - n_done);
            out += "<\xC2\xA6";
            out.append(text, i + 2, j - i - 2);
            out += "\xC2\xA6>";
            n_done = j + 2;
        }
    }
    if (n_done == 0) {
        return text;
    }
    out.append(text, n_done, std::string::npos);
    return out;
}


DecisionContext::DecisionContext(DecisionModelType modelType) : 
	modelType(modelType)
{
}

bool DecisionContext::IsJointTask() const
{
	return false;
}

//
// model-specific setup
//

std::unique_ptr<DecisionContext> DecisionContext::Create(const llama_model* model)
{
	const std::string prefix = DecisionContext::meta_str(model, "general.architecture") + ".decision.";
	const std::string type_name = DecisionContext::meta_str(model, prefix + "type");

	std::unique_ptr<DecisionContext> dcx;

	if (type_name == "laya")
	{
		dcx = std::make_unique<LayaDecisionContext>();
	}
	else if (type_name == "lev")
	{
		dcx = std::make_unique<LevDecisionContext>();
	}
	else if (type_name == "nimble")
	{
		dcx = std::make_unique<NimbleDecisionContext>();
	}
	else if (type_name == "clef")
	{
		dcx = std::make_unique<ClefDecisionContext>();
	}
	else if (type_name == "openjev")
	{
		dcx = std::make_unique<OpenJevDecisionContext>();
	}
	else if (type_name == "kev")
	{
		dcx = std::make_unique<KevDecisionContext>();
	}
	else if (type_name == "granite-switch" || type_name == "granite_switch")
	{
		dcx = std::make_unique<GraniteSwitchDecisionContext>();
	}
	else if (type_name == "pplx-decider" || type_name == "pplx_decider")
	{
		dcx = std::make_unique<PplxDeciderDecisionContext>();
	}
	else if (type_name == "vllm-decision2" || type_name == "vllm_decision2")
	{
		dcx = std::make_unique<VllmDecision2DecisionContext>();
	}
	else if (type_name == "vllm-d3" || type_name == "vllm_d3")
	{
		dcx = std::make_unique<VllmD3DecisionContext>();
	}
	else if (type_name == "lfm2-d1" || type_name == "lfm2_d1")
	{
		dcx = std::make_unique<Lfm2D1DecisionContext>();
	}
	else if (type_name == "lfm2-d1-omni" || type_name == "lfm2_d1_omni")
	{
		dcx = std::make_unique<Lfm2D1OmniDecisionContext>();
	}
	else
	{			
		if (type_name.empty())
		{
			return nullptr;
		}
		else
		{
			MY_LOG_ERROR("unsupported decision model type: %s", type_name.c_str());
		}

		dcx = std::unique_ptr<DecisionContext>(new DecisionContext(DecisionModelType::UNKNOWN));
	}
	dcx->Init(model, prefix);

	return dcx;
}

void DecisionContext::Init(const llama_model* model, const std::string& prefix)
{
	const char* tpl = llama_model_chat_template(model, "systemone");
	if (tpl == nullptr)
	{
		MY_LOG_WARNING("decision model has no \"systemone\" template");
	}
	else
	{
		this->tpl = std::make_shared<DecisionTemplate>(tpl);
	}
		
	vocab = llama_model_get_vocab(model);
	
	char key[256];
	char val[64];

	const std::string prefix_temp = prefix + "temperature.";
	for (int32_t i = 0; i < llama_model_meta_count(model); i++)
	{		
		if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !DecisionContext::starts_with(key, prefix_temp))
		{
			continue;
		}

		if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0)
		{
			continue;
		}

		const float temp = std::strtof(val, nullptr);
		if (!std::isfinite(temp) || temp <= 0.0f)
		{
			throw std::runtime_error(std::string("invalid decision temperature: ") + key + " = " + val);
		}

		temperatures[key + prefix_temp.size()] = temp;
	}



    // V2 calibration offsets are indexed by number of levels and level index.
    const std::string prefix_bias = prefix + "score_bias.";
    for (int32_t i = 0; i < llama_model_meta_count(model); ++i) 
	{        
		if (llama_model_meta_key_by_index(model, i, key, sizeof(key)) < 0 || !starts_with(key, prefix_bias))
		{
			continue;
		}

		if (llama_model_meta_val_str_by_index(model, i, val, sizeof(val)) < 0)
		{
			continue;
		}

        const auto parts = split(key + prefix_bias.size(), ".");
        const size_t n = parts.size() == 2 ? std::strtoul(parts[0].c_str(), nullptr, 10) : 0;
        const size_t level = parts.size() == 2 ? std::strtoul(parts[1].c_str(), nullptr, 10) : 0;
        const float value = std::strtof(val, nullptr);

        if (n < 2 || n > 255 || level >= n || !std::isfinite(value)) 
		{
            MY_LOG_ERROR("invalid decision score bias: %s", key);
            continue;
        }

        auto& bias = score_bias[n];
        bias.resize(n, 0.0f);
        bias[level] = value;
    }


	this->InitSetup(model, prefix);				

}


//
// request parsing
//

std::vector<PreparedQuestion> DecisionContext::PrepareQuestions(const std::vector<Question>& input) const
{
    // The struct API already guarantees text payloads. Decision2 additionally
    // requires non-empty instructions/keys and explicit NOUL descriptions.
    for (const auto& question : input)
	{
        if (question.useDefaultInstructions && modelType != DecisionModelType::VLLM_D3) 
		{
            MY_LOG_ERROR("default instructions are supported only by VLLM-D3");
            return {};
        }

		if (!strict_payload)
		{
			continue;
		}

        if (question.instructions.empty()) 
		{
            MY_LOG_ERROR("instructions must be non-empty text");
            return {};
        }

        if (question.type == QuestionType::CHOICE && question.criteria.size() < 2) 
		{
            MY_LOG_ERROR("choice criteria must contain at least two options");
            return {};
        }

        for (const auto& option : question.criteria) 
		{
            if ((question.type == QuestionType::CHOICE && option.key.empty()) ||
                (question.type == QuestionType::NOUL &&
                    ((option.key != "false" && option.key != "true") || !option.value))) 
			{
                MY_LOG_ERROR("invalid decision criteria payload");
                return {};
            }
        }
    }

	std::set<std::string> ids;
	std::vector<PreparedQuestion> result;

	for (const auto& question : input)
	{
		if (ids.insert(question.id).second == false)
		{
			MY_LOG_INFO("duplicate question id: %s", question.id.c_str());
			continue;
		}

		PreparedQuestion q;
		static_cast<Question&>(q) = question;

		for (size_t i = 0; i < question.criteria.size(); ++i)
		{
			q.originalIndices.push_back(i);
		}

		if (q.type == QuestionType::CHOICE)
		{
			if (q.criteria.empty())
			{
				throw std::invalid_argument("choice requires at least one criterion");
			}

			std::set<std::string> keys;
			for (const auto& option : q.criteria)
			{
				if (keys.insert(option.key).second == false)
				{
					MY_LOG_WARNING("duplicate choice key: %s", option.key.c_str());
					continue;
				}
			}

			if (this->modelType == DecisionModelType::CLEF) 
			{
				std::sort(q.originalIndices.begin(), q.originalIndices.end(), [&](size_t a, size_t b) {
					return question.criteria[a].key < question.criteria[b].key;
				});

				for (size_t i = 0; i < q.criteria.size(); ++i)
				{
					q.criteria[i] = question.criteria[q.originalIndices[i]];
				}
			}
		}
		else if (q.type == QuestionType::SCORE)
		{
			if (q.criteria.size() < 2 || q.criteria.size() > 10)
			{
				throw std::invalid_argument("score requires 2 to 10 levels");
			}

			for (size_t i = 0; i < q.criteria.size(); ++i) 
			{
				// Key-only criteria are level descriptions; model keys are indices.
				q.criteria[i] = { 
					std::to_string(i), 
					question.criteria[i].value.value_or(question.criteria[i].key) 
				};
			}
		}
		else if (q.type == QuestionType::NOUL)
		{
			q.criteria = { 
				{"false", std::nullopt}, 
				{"true", std::nullopt} 
			};

			constexpr size_t absent = std::numeric_limits<size_t>::max();
			q.originalIndices = { absent, absent };
			std::set<std::string> keys;

			for (size_t i = 0; i < question.criteria.size(); ++i)
			{
				const auto& option = question.criteria[i];
                const std::string key = (allow_noul_aliases && option.key == "yes") ? "true" :
                    (allow_noul_aliases && option.key == "no") ? "false" : option.key;

				if ((key != "false" && key != "true") || !keys.insert(key).second)
				{
					throw std::invalid_argument("noul criteria must have unique false/true keys");
				}

				const size_t index = key == "true" ? 1 : 0;
				q.criteria[index] = {key, option.value};
				q.originalIndices[index] = i;
			}

			if (question.criteria.empty())
			{ 				
				q.originalIndices = { 0, 1 }; 
			}

			if (noul_true_first || (noul_criteria_order && question.criteria.size() == 2 && question.criteria[0].key == "true"))
			{
				std::swap(q.criteria[0], q.criteria[1]);
				std::swap(q.originalIndices[0], q.originalIndices[1]);
			}
		}

		if (q.criteria.size() > n_options_max)
		{
			throw std::invalid_argument("too many decision criteria");
		}

		result.push_back(std::move(q));
	}

	return result;
}

size_t DecisionContext::GetVariantsCount(const PreparedQuestion& question) const
{	
	return 1;
}

size_t DecisionContext::GetOutputsCount(const PreparedQuestion& question) const
{	
	return question.criteria.size();
}


std::string DecisionContext::RunTemplate(const PromptInput& input) const
{
    return (this->customPromptBuilder) ? this->customPromptBuilder(input) : this->tpl->Run(input);    
}

bool DecisionContext::TokenizePrompt(const std::string& prompt, DecisionPrompt& task) const
{   
    task.tokens = tokenize(vocab, prompt, false, true);

    return true;
}

std::string DecisionContext::Render(const std::string& state,
	const std::vector<PreparedQuestion>& questions, const PreparedQuestion& question, size_t variant) const
{
	PromptInput input;
	input.modelType = modelType;
	input.state = state;
    input.stateIsNull = state_is_null;	
	input.variant = variant;
	input.question = question;

	if (variant != 0)
	{
		std::reverse(input.question.criteria.begin(), input.question.criteria.end());
	}		
	
	auto n = this->GetOutputsCount(question);

	if (!label_texts.empty() && n > label_texts.size())
	{
		throw std::runtime_error("decision model has too few label tokens");
	}

	if (!label_texts.empty())
	{
		input.labels.assign(label_texts.begin(), label_texts.begin() + n);
	}

    
	if (modelType == DecisionModelType::NIMBLE)
	{
		for (const auto& q : questions)
		{
			input.questions.push_back(q);
			input.questionLabels.emplace_back(label_texts.begin(), label_texts.begin() + this->GetOutputsCount(q));
		}
	}

	this->PreparePrompt(input);

	return this->RunTemplate(input);
}


void DecisionContext::FillTask(
	const std::string& state,
	const std::vector<PreparedQuestion>& questions,
	const PreparedQuestion& question,
	size_t variant,
	DecisionPrompt& task) const
{
	const std::string prompt = this->Render(state, questions, question, variant);

	
    if (!this->TokenizePrompt(prompt, task)) {
        task = {};
        return;
    }
	std::vector<llama_token> tokens = std::move(task.tokens);
	if (tokens.empty()) 
	{
		throw std::invalid_argument("decision prompt must not be empty");
	}

	this->FillModelTak(tokens, question, task);	

	task.tokens = std::move(tokens);
}




//
// answer
//

// confidence formulas are the ones published by TypeSafe

static double decision_confidence_choice(const std::vector<double>& probs) 
{
	if (probs.size() < 2) 
	{
		return 1.0;
	}
	const double uniform = 1.0 / probs.size();
	const double p_max = *std::max_element(probs.begin(), probs.end());
	return std::max(0.0, (p_max - uniform) / (1.0 - uniform));
}

static double decision_confidence_score(const std::vector<double>& probs) 
{
	if (probs.size() < 2) 
	{
		return 1.0;
	}
	const size_t n = probs.size();
	const size_t mode = std::max_element(probs.begin(), probs.end()) - probs.begin();

	// mean distance to the mode, relative to the one of a uniform distribution around its center
	double dist = 0.0;
	double dist_uniform = 0.0;
	for (size_t i = 0; i < n; i++) 
	{
		dist += probs[i] * std::fabs((double)i - (double)mode);
		dist_uniform += std::fabs((double)i - (n - 1) / 2.0) / n;
	}
	return std::max(0.0, 1.0 - dist / dist_uniform);
}


static double decision_confidence_entropy(const std::vector<double>& probabilities)
{
    if (probabilities.size() < 2) return 1.0;
    double entropy = 0.0;
    for (const double p : probabilities) if (p > 0.0) entropy -= p * std::log(p);
    return std::max(0.0, std::min(1.0, 1.0 - entropy / std::log(static_cast<double>(probabilities.size()))));
}

bool DecisionContext::SetNullState(bool value)
{
    if (value && !allow_null_state) 
	{
        MY_LOG_ERROR("this decision model requires a non-null state");
        return false;
    }
    state_is_null = value;
    return true;
}

std::vector<float> DecisionContext::AggregateLabelScores(const std::vector<float>& scores,
    const std::vector<size_t>& groups)
{
    if (groups.empty()) return scores;
    std::vector<float> result;
    size_t offset = 0;
    for (const size_t count : groups) {
        if (count == 0 || count > scores.size() - offset) {
            MY_LOG_ERROR("invalid decision label group size");
            return {};
        }
        const auto begin = scores.begin() + offset;
        const float maximum = *std::max_element(begin, begin + count);
        double sum = 0.0;
        if (std::isfinite(maximum)) {
            for (size_t i = 0; i < count; ++i) sum += std::exp(static_cast<double>(scores[offset + i]) - maximum);
        }
        result.push_back(std::isfinite(maximum) ? static_cast<float>(maximum + std::log(sum)) : maximum);
        offset += count;
    }
    if (offset != scores.size()) {
        MY_LOG_ERROR("decision label groups do not cover the logits");
        return {};
    }
    return result;
}

DecisionAnswer DecisionContext::FormatAnswer(const PreparedQuestion& question, const std::vector<std::vector<float>>& scores) const
{
	const size_t n = this->GetOutputsCount(question);
	if (scores.size() != this->GetVariantsCount(question))
	{
		throw std::runtime_error("decision result does not match the number of variants");
	}

	// softmax over the outputs of each variant, then the average of the variants
	const float temperature = this->GetTemperature(question);

	std::vector<double> probs(n, 0.0);

	for (size_t v = 0; v < scores.size(); v++) 
	{
		const auto& s = scores[v];
		if (s.size() != n) 
		{
			throw std::runtime_error("decision result does not match the number of options");
		}

		// a joint head returns NaN if it could not use the decision order
		if (std::any_of(s.begin(), s.end(), [](float v) { return !std::isfinite(v); })) 
		{
			throw std::runtime_error("the model could not evaluate the decision");
		}

        std::vector<double> logits(s.begin(), s.end());
        if (question.type == QuestionType::SCORE && score_bias.count(n)) 
		{
			for (size_t i = 0; i < n; ++i)
			{
				logits[i] += score_bias.at(n)[i];
			}
        }

        const double score_max = *std::max_element(logits.begin(), logits.end());
		std::vector<double> p(n);
		double sum = 0.0;

		for (size_t i = 0; i < n; i++)
		{
			p[i] = std::exp((logits[i] - score_max) / temperature);
			sum += p[i];
		}
		for (size_t i = 0; i < n; i++) 
		{
			// the second variant is in the reverse order
			probs[v == 0 ? i : n - 1 - i] += p[i] / sum / scores.size();
		}
	}

	DecisionAnswer answer;
	answer.id = question.id;
	answer.probabs.resize(question.criteria.empty() ? 2 : question.criteria.size());

	if (question.type == QuestionType::NOUL) 
	{
		double yes = 0.0;
		if (modelType == DecisionModelType::LEV) 
		{
			for (size_t i = 0; i < n; ++i)
			{
				yes += probs[i] * i / (n - 1);
			}
		}
		else 
		{
			for (size_t i = 0; i < n; ++i)
			{
				if (question.criteria[i].key == "true") yes = probs[i];
			}
		}

		answer.confidence = static_cast<float>(yes);

		for (size_t i = 0; i < question.criteria.size(); ++i) 
		{
			if (question.originalIndices[i] < answer.probabs.size())
			{
				answer.probabs[question.originalIndices[i]] = static_cast<float>(question.criteria[i].key == "true" ? yes : 1.0 - yes);
			}
		}
	}
	else 
	{
		answer.confidence = static_cast<float>(entropy_confidence ? decision_confidence_entropy(probs) :
            question.type == QuestionType::CHOICE ? decision_confidence_choice(probs) : decision_confidence_score(probs));

		for (size_t i = 0; i < n; ++i)
		{
			answer.probabs.at(question.originalIndices[i]) = static_cast<float>(probs[i]);
		}
	}

	return answer;
}

float DecisionContext::GetTemperature(const PreparedQuestion& question) const 
{
	const size_t n = question.criteria.size();
	const std::string type_name = QuestionTypeToString(question.type);

	// the temperature can depend on the number of options, the buckets are the ones used to fit it
	std::string bucket;
	if (modelType == DecisionModelType::LEV) 
	{
		bucket = n <= 8 ? "small" : n <= 26 ? "mid" : "large";
	}	
	else 
	{
		bucket = n <= 2 ? "2" : n <= 5 ? "3_5" : n <= 10 ? "6_10" : "11";
	}

	for (const auto& name : { type_name + "." + bucket, type_name }) 
	{
		const auto it = temperatures.find(name);
		if (it != temperatures.end()) 
		{
			return it->second;
		}
	}	

	return 1.0f;
}

