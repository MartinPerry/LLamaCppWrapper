#include "./LLamaDecisionVllmDecision2.h"

#if __has_include (<llama.h>)

VllmDecision2DecisionContext::VllmDecision2DecisionContext() : 
    ClefDecisionContext(DecisionModelType::VLLM_DECISION2) 
{
}

void VllmDecision2DecisionContext::InitSetup(const llama_model*, const std::string&)
{
    n_options_max = 255;
    noul_criteria_order = true;
    strict_payload = true;
    entropy_confidence = true;
}

bool VllmDecision2DecisionContext::IsJointTask() const 
{
    return false; 
}

void VllmDecision2DecisionContext::FillTask(const std::string& state, const std::vector<PreparedQuestion>&,
    const PreparedQuestion& question, size_t variant, DecisionPrompt& task) const
{
    // The head is joint, but V2 evaluates one question per prompt for this model.
    ClefDecisionContext::FillTask(state, {question}, question, variant, task);
}

#endif
