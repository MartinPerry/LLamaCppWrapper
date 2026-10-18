#include "./LLamaDecisionPplxDecider.h"

#if __has_include (<llama.h>)

PplxDeciderDecisionContext::PplxDeciderDecisionContext() : 
    CodeLabelsDecisionContext(DecisionModelType::PPLX_DECIDER) 
{    
}

bool PplxDeciderDecisionContext::SupportsMedia() const
{
    return true;
}

#endif
