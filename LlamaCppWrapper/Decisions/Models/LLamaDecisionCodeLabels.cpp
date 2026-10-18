#include "./LLamaDecisionCodeLabels.h"

#if __has_include (<llama.h>)

#include <stdexcept>

CodeLabelsDecisionContext::CodeLabelsDecisionContext(DecisionModelType type):
	DecisionContext(type)
{
}

void CodeLabelsDecisionContext::InitSetup(const llama_model* model, const std::string& prefix)
{
	// label codes are A..Z then AA..ZZ, only the ones that are a single token are used
	std::vector<std::string> codes;
	for (char a = 'A'; a <= 'Z'; a++)
	{
		codes.push_back(std::string(1, a));
	}
	for (char a = 'A'; a <= 'Z'; a++)
	{
		for (char b = 'A'; b <= 'Z'; b++)
		{
			codes.push_back(std::string{ a, b });
		}
	}
	for (const auto& code : codes)
	{
		const auto toks = DecisionContext::tokenize(vocab, code, false, false);
		if (toks.size() == 1 && labels.size() < 255)
		{
			labels.push_back(toks[0]);
			label_texts.push_back(code);
		}
	}
	n_options_max = labels.size();
}

void CodeLabelsDecisionContext::FillModelTak(std::vector<llama_token>& tokens,
	const PreparedQuestion& question, DecisionPrompt& task) const
{
	auto n = this->GetOutputsCount(question);
	
	//reads one label logit per option.
	if (n > labels.size())
	{
		throw std::runtime_error("decision model has too few label tokens");
	}

	task.labels.assign(labels.begin(), labels.begin() + n);
}

#endif
