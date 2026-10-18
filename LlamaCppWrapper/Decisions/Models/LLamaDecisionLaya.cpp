#include "./LLamaDecisionLaya.h"

#if __has_include (<llama.h>)

#include <stdexcept>
#include <algorithm>
#include <cstdlib>


LayaDecisionContext::LayaDecisionContext() : 
	DecisionContext(DecisionModelType::LAYA)
{
}

void LayaDecisionContext::InitSetup(const llama_model* model, const std::string& prefix)
{
	token_marker = llama_vocab_mask(vocab);
	token_sep = llama_vocab_sep(vocab);
	if (token_marker == LLAMA_TOKEN_NULL || token_sep == LLAMA_TOKEN_NULL)
	{
		throw std::runtime_error("decision model has no mask or sep token");
	}
	text_marker = DecisionContext::token_to_piece(vocab, token_marker);

	const std::string val = DecisionContext::meta_str(model, prefix + "max_head_tokens");
	max_head_tokens = std::strtoul(val.c_str(), nullptr, 10);
	
	if (max_head_tokens == 0)
	{
		throw std::runtime_error("decision model has no valid max_head_tokens");
	}

	n_options_max = 255;
}

// the prompt is: [cls] question [sep] ([marker] option)* [sep] state [sep]
// options and question are cut to fit max_head_tokens, the same way the model was trained
void LayaDecisionContext::FillModelTak(std::vector<llama_token>& tokens,
	const PreparedQuestion& question, DecisionPrompt& task) const
{
	const size_t n_options = question.criteria.size();

	std::vector<size_t> markers;
	for (size_t i = 0; i < tokens.size(); i++) 
	{
		if (tokens[i] == token_marker) 
		{
			markers.push_back(i);
		}
	}

	const auto invalid = std::runtime_error("unexpected layout of the decision prompt");
	if (markers.size() != n_options || markers[0] < 2 || tokens[markers[0] - 1] != token_sep || tokens.back() != token_sep) 
	{
		throw invalid;
	}

	const size_t head_end = markers[0] - 1;
	const size_t opts_end = std::find(tokens.begin() + markers.back(), tokens.end(), token_sep) - tokens.begin();

	if (opts_end + 1 >= tokens.size()) 
	{
		throw invalid;
	}

	// marker + text of each option
	std::vector<std::vector<llama_token>> options;
	size_t n_options_tokens = 0;

	auto set_max = [&](size_t n_max) {
		n_options_tokens = 0;
		for (auto& opt : options) 
		{
			opt.resize(std::min(opt.size(), n_max));
			n_options_tokens += opt.size();
		}
	};

	for (size_t i = 0; i < n_options; i++) 
	{
		const size_t end = i + 1 < n_options ? markers[i + 1] : opts_end;
		options.emplace_back(tokens.begin() + markers[i], tokens.begin() + end);
	}

	set_max(max_option_tokens + 1);

	if (n_options_tokens + 16 > max_head_tokens) 
	{
		// too many or too long options, shrink them evenly
		set_max(std::max((size_t)4, (max_head_tokens - std::min(max_head_tokens, (size_t)16)) / n_options));
	}

	const size_t n_question_max = std::max((size_t)8, max_head_tokens - std::min(max_head_tokens, n_options_tokens));

	std::vector<llama_token> out;
	out.push_back(tokens[0]);
	out.insert(out.end(), tokens.begin() + 1, tokens.begin() + std::min(head_end, 1 + n_question_max));
	out.push_back(token_sep);

	for (const auto& opt : options) 
	{
		task.markers.push_back(static_cast<int32_t>(out.size()));
		out.insert(out.end(), opt.begin(), opt.end());	
	}

	out.insert(out.end(), tokens.begin() + opts_end, tokens.end());
	tokens = std::move(out);

	// the output has one score per question type
	task.column = static_cast<int32_t>(question.type);
}


void LayaDecisionContext::PreparePrompt(PromptInput& input) const
{	
	DecisionContext::replace_all(input.question.id, text_marker, " ");
	DecisionContext::replace_all(input.question.instructions, text_marker, " ");
	
	for (auto& option : input.question.criteria)
	{
		DecisionContext::replace_all(option.key, text_marker, " ");		
		if (option.value)
		{
			DecisionContext::replace_all(*option.value, text_marker, " ");			
		}
	}
	

	if (text_marker.empty() == false)
	{
		DecisionContext::replace_all(input.state, text_marker, " ");
	}
}

#endif
