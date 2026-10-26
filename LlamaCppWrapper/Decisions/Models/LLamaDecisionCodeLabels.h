#ifndef LLAMA_DECISION_CODE_LABELS_H
#define LLAMA_DECISION_CODE_LABELS_H

#if __has_include (<llama.h>)

#include <stdexcept>
#include <string>
#include <vector>

#include <llama.h>

#include "../LLamaDecisionStructures.h"
#include "../LLamaDecisionContext.h"
#include "../LLamaDecisionContextMultiModal.h"

// Shared A..Z / AA..ZZ label setup for the new V2 label models.
template <class Parent>
class CodeLabelsDecisionContext : public Parent
{
protected:
    CodeLabelsDecisionContext(DecisionModelType type);

    void InitSetup(const llama_model* model, const std::string& prefix) override;
    void FillModelTak(std::vector<llama_token>& tokens, const PreparedQuestion& question, DecisionPrompt& task) const override;
};

//================================================================

template <class Parent>
CodeLabelsDecisionContext<Parent>::CodeLabelsDecisionContext(DecisionModelType type) :
	Parent(type)
{
}

template <class Parent>
void CodeLabelsDecisionContext<Parent>::InitSetup(const llama_model* model, const std::string& prefix)
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
		const auto toks = DecisionContext::tokenize(this->vocab, code, false, false);
		if (toks.size() == 1 && this->labels.size() < 255)
		{
			this->labels.push_back(toks[0]);
			this->label_texts.push_back(code);
		}
	}
	this->n_options_max = this->labels.size();
}

template <class Parent>
void CodeLabelsDecisionContext<Parent>::FillModelTak(std::vector<llama_token>& tokens,
	const PreparedQuestion& question, DecisionPrompt& task) const
{
	auto n = this->GetOutputsCount(question);

	//reads one label logit per option.
	if (n > this->labels.size())
	{
		throw std::runtime_error("decision model has too few label tokens");
	}

	task.labels.assign(this->labels.begin(), this->labels.begin() + n);
}


#endif
#endif
