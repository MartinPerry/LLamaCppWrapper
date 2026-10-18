#ifndef LLAMA_DECISION_TEMPLATE_H
#define LLAMA_DECISION_TEMPLATE_H

#if __has_include (<llama.h>)

namespace jinja
{
	struct program;
};

class common_json;

#include <vector>
#include <string>
#include <memory>

#include "./LLamaDecisionStructures.h"

class DecisionTemplate 
{
public:
	DecisionTemplate(const std::string& source);
	~DecisionTemplate() = default;
	
	std::string Run(const PromptInput& input) const;

protected:

	std::shared_ptr<const jinja::program> program;
	std::string source;

	common_json SortKeys(const common_json& value) const;
	common_json Input(const PromptInput& input) const;
	common_json ToJson(const Question& question, const std::vector<std::string>& labels) const;

};

#endif
#endif