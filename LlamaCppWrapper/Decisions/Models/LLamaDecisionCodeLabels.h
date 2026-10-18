#ifndef LLAMA_DECISION_CODE_LABELS_H
#define LLAMA_DECISION_CODE_LABELS_H

#if __has_include (<llama.h>)

#include "../LLamaDecisionContext.h"

// Shared A..Z / AA..ZZ label setup for the new V2 label models.
class CodeLabelsDecisionContext : public DecisionContext 
{
protected:
    CodeLabelsDecisionContext(DecisionModelType type);

    void InitSetup(const llama_model* model, const std::string& prefix) override;
    void FillModelTak(std::vector<llama_token>& tokens, const PreparedQuestion& question, DecisionPrompt& task) const override;
};

#endif
#endif
