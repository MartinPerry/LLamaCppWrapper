#ifndef LLAMA_DECISION_PPLX_DECIDER_H
#define LLAMA_DECISION_PPLX_DECIDER_H

#if __has_include (<llama.h>)

#include "./LLamaDecisionCodeLabels.h"

class PplxDeciderDecisionContext : public CodeLabelsDecisionContext<DecisionContextMultiModal>
{
public:
    PplxDeciderDecisionContext();
    
};

#endif
#endif
