#include "./LLamaDecisionKev.h"

#if __has_include (<llama.h>)

#include <stdexcept>
#include <algorithm>
#include <cstdlib>


KevDecisionContext::KevDecisionContext() :
	DecisionContext(DecisionModelType::KEV)
{
}

void KevDecisionContext::InitSetup(const llama_model* model, const std::string& prefix)
{
	// the hidden state of an option is read at the token that ends it
	const auto toks = DecisionContext::tokenize(vocab, "<|box_end|>", false, true);
	if (toks.size() != 1)
	{
		throw std::runtime_error("decision model has no <|box_end|> token");
	}

	token_marker = toks[0];
	n_options_max = 255;
}

// The pointer head reads the question at the final token and each option at its end marker.
void KevDecisionContext::FillModelTak(std::vector<llama_token>& tokens,
	const PreparedQuestion& question, DecisionPrompt& task) const
{
	// an option is read at its end token, the question at the last token
	for (size_t i = 0; i < tokens.size(); i++) 
	{
		if (tokens[i] == token_marker) 
		{
			task.markers.push_back(static_cast<int32_t>(i));
		}
	}

	if (task.markers.size() != question.criteria.size()) 
	{
		throw std::runtime_error("unexpected layout of the decision prompt");
	}

	task.pointer = static_cast<int32_t>(tokens.size() - 1);
}

void KevDecisionContext::PreparePrompt(PromptInput& input) const
{

	input.state = escape_special_tokens(input.state);

	escape_special_tokens(input.question.id);
	escape_special_tokens(input.question.instructions);
	for (auto& option : input.question.criteria)
	{
		escape_special_tokens(option.key);
		if (option.value)
		{
			escape_special_tokens(*option.value);
		}
	}

}

#endif
