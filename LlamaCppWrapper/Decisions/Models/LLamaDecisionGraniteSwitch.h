#ifndef LLAMA_DECISION_GRANITE_SWITCH_H
#define LLAMA_DECISION_GRANITE_SWITCH_H

#if __has_include (<llama.h>)

#include "./LLamaDecisionCodeLabels.h"

class GraniteSwitchDecisionContext : public CodeLabelsDecisionContext<DecisionContext>
{
public:
    GraniteSwitchDecisionContext();

protected:
    void InitSetup(const llama_model* model, const std::string& prefix) override;

};

#endif
#endif
