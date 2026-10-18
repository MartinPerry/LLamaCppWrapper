#include "./LLamaDecisionLev.h"

#if __has_include (<llama.h>)

#include <stdexcept>
#include <algorithm>
#include <cstdlib>

#include "../LLamaDecisionContext.h"

LevDecisionContext::LevDecisionContext() :
	CodeLabelsDecisionContext(DecisionModelType::LEV)
{
}


size_t LevDecisionContext::GetVariantsCount(const PreparedQuestion& question) const
{
	// lev shows the options of a choice in 2 orders, to cancel the preference for the first label
	if (question.type == QuestionType::CHOICE && question.criteria.size() > 1)
	{
		return 2;
	}

	return DecisionContext::GetVariantsCount(question);
}

size_t LevDecisionContext::GetOutputsCount(const PreparedQuestion& question) const
{
	if (question.type == QuestionType::NOUL)
	{
		// lev reads noul from a rating scale: 0 = certainly no, 8 = certainly yes		
		return 9;
	}

	return DecisionContext::GetOutputsCount(question);
}

#endif