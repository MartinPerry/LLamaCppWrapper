#include "./LLamaDecisionVllmD3.h"

#if __has_include (<llama.h>)

VllmD3DecisionContext::VllmD3DecisionContext() : 
    CodeLabelsDecisionContext(DecisionModelType::VLLM_D3) 
{
}

void VllmD3DecisionContext::InitSetup(const llama_model* model, const std::string& prefix)
{
    CodeLabelsDecisionContext::InitSetup(model, prefix);    
    entropy_confidence = true; 
    allow_null_state = true;    
}

#endif
