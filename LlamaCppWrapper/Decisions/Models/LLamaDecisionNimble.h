#ifndef LLAMA_DECISION_NIMBLE_H
#define LLAMA_DECISION_NIMBLE_H

#if __has_include (<llama.h>)

#include "./LLamaDecisionCodeLabels.h"

class NimbleDecisionContext : public CodeLabelsDecisionContext<DecisionContext>
{
public:
    NimbleDecisionContext();


};

#endif
#endif
