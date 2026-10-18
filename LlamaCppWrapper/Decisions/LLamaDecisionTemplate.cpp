#include "./LlamaDecisionTemplate.h"

#include "../3rdParty/json.h"
#include "../3rdParty/jinja/jparser.h"

#include "../Utils/Logger.h"

#include "./LLamaDecision.h"

DecisionTemplate::DecisionTemplate(const std::string& source) : 
	source(source)
{	
	this->program = std::make_shared<const jinja::program>(
		jinja::parse_from_tokens(jinja::lexer().tokenize(source))
	);
}


std::string DecisionTemplate::Run(const PromptInput& input) const
{		
	const common_json variables = this->Input(input);
	jinja::context context(source);
	jinja::global_from_json(context, variables, false);
	jinja::runtime runtime(context);
	const jinja::value result = runtime.execute(*program);
	return jinja::runtime::gather_string_parts(result)->as_string().str();
}

// Keep JSON conversion private to the default template builder. Public inputs
// and answers continue to use the application's structs.
common_json DecisionTemplate::SortKeys(const common_json& value) const
{
	if (value.is_array()) 
	{
		common_json result = common_json::array();
		for (const auto& item : value)
		{
			result.push_back(this->SortKeys(item));
		}

		return result;
	}

	if (value.is_object()) 
	{
		std::map<std::string, common_json> sorted;
		for (const auto& [key, item] : value.items())
		{
			sorted.emplace(key, this->SortKeys(item));
		}

		common_json result = common_json::object();
		for (const auto& [key, item] : sorted)
		{
			result[key] = item;
		}

		return result;
	}
	return value;
}


common_json DecisionTemplate::ToJson(const Question& question, const std::vector<std::string>& labels) const
{
	common_json options = common_json::array();
	if (!labels.empty() && labels.size() < question.criteria.size()) 
	{
		throw std::invalid_argument("too few decision labels for the template");
	}

	for (size_t i = 0; i < question.criteria.size(); ++i) 
	{
		const auto& option = question.criteria[i];
		common_json item = {
			{"key", option.key},
			{"description", option.value ? common_json(*option.value) : common_json()},
		};

		if (!labels.empty())
		{
			item["label"] = labels[i];
		}

		options.push_back(std::move(item));
	}

	return common_json {
		{"id", question.id},
		{"type", QuestionTypeToString(question.type)},
		{"instructions", question.useDefaultInstructions ? common_json() : common_json(question.instructions)},
		{"options", options},
	};
};

common_json DecisionTemplate::Input(const PromptInput& input) const
{
	
	common_json result;
	if (input.joint) 
	{
		result = common_json{ 
			{"state", input.stateIsNull ? common_json() : common_json(input.state)}, 
			{"questions", common_json::array()} 
		};

		for (const auto& question : input.questions) 
		{
			result["questions"].push_back(this->ToJson(question, {}));
		}

		result = this->SortKeys(result);
		result["images"] = common_json(input.images);
		result["sep"] = input.separator;
		result["mark_question"] = input.questionMarker;
		result["mark_option"] = input.optionMarker;
	}
	else 
	{
		const common_json question = this->ToJson(input.question, input.labels);
		result = common_json {
			{"id", input.question.id},
			{"type", QuestionTypeToString(input.question.type)},
			{"instructions", question.at("instructions")},
			{"state", input.stateIsNull ? common_json() : common_json(input.state)},
			{"options", question.at("options")},
		};

		if (input.modelType == DecisionModelType::NIMBLE) 
		{
			if (input.questionLabels.size() != input.questions.size()) 
			{
				throw std::invalid_argument("decision question labels do not match the questions");
			}

			result["questions"] = common_json::array();
			for (size_t i = 0; i < input.questions.size(); ++i) 
			{
				result["questions"].push_back(this->ToJson(input.questions[i], input.questionLabels[i]));
			}
		}
		else if (input.modelType == DecisionModelType::LEV)
		{
			result = this->SortKeys(result);
		}
        else if (input.modelType == DecisionModelType::LFM2_D1_OMNI) 
		{
            result["audio"] = input.audio;
            result["sep"] = input.separator;
            result["mark_state"] = input.stateMarker;
            result["mark_question"] = input.questionMarker;
            result["mark_option"] = input.optionMarker;
        }

		result["images"] = common_json(input.images);
	}
	return result;
}
