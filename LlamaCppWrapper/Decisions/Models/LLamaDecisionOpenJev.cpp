#include "./LLamaDecisionOpenJev.h"

#if __has_include (<llama.h>)

#include <stdexcept>
#include <algorithm>
#include <cstdlib>


OpenJevDecisionContext::OpenJevDecisionContext() :
	DecisionContext(DecisionModelType::OPENJEV)
{
}

bool OpenJevDecisionContext::SupportsMedia() const
{
	return true;
}

void OpenJevDecisionContext::InitSetup(const llama_model* model, const std::string& prefix)
{    
	// one letter per option, each must be a single token
	const std::string letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	for (const char c : letters)
	{
		const auto toks = DecisionContext::tokenize(vocab, std::string(1, c), false, false);
		if (toks.size() != 1)
		{
			throw std::runtime_error(std::string("decision label is not a single token: ") + c);
		}
		labels.push_back(toks[0]);
	}
	n_options_max = labels.size();
	noul_true_first = true;
	
}

// Read the option scores from the label logits at the last prompt token.
void OpenJevDecisionContext::FillModelTak(std::vector<llama_token>& tokens,
	const PreparedQuestion& question, DecisionPrompt& task) const
{
	auto n = this->GetOutputsCount(question);
	if (n > labels.size())
	{
		throw std::runtime_error("decision model has too few label tokens");
	}
	task.labels.assign(labels.begin(), labels.begin() + n);
}

#endif
