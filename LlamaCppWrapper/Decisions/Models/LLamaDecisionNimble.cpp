#include "./LLamaDecisionNimble.h"

#if __has_include (<llama.h>)

#include <stdexcept>
#include <algorithm>
#include <cstdlib>


NimbleDecisionContext::NimbleDecisionContext() :
	CodeLabelsDecisionContext(DecisionModelType::NIMBLE)
{
}



#endif
