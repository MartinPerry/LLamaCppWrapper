#include "./LLamaDecisionGraniteSwitch.h"

#if __has_include (<llama.h>)

GraniteSwitchDecisionContext::GraniteSwitchDecisionContext() : 
    CodeLabelsDecisionContext(DecisionModelType::GRANITE_SWITCH) 
{
}

void GraniteSwitchDecisionContext::InitSetup(const llama_model* model, const std::string& prefix)
{
    CodeLabelsDecisionContext::InitSetup(model, prefix);
    noul_true_first = true;
}

#endif
