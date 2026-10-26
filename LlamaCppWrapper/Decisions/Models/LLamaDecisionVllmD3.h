#ifndef LLAMA_DECISION_VLLM_D3_H
#define LLAMA_DECISION_VLLM_D3_H

#if __has_include (<llama.h>)

#include "./LLamaDecisionCodeLabels.h"

class VllmD3DecisionContext : public CodeLabelsDecisionContext<DecisionContextMultiModal>
{
public:
    VllmD3DecisionContext();
    
protected:

    void InitSetup(const llama_model* model, const std::string& prefix) override;
};

#endif
#endif
