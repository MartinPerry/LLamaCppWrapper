#ifndef LLAMA_DECISION_KEV_H
#define LLAMA_DECISION_KEV_H

#if __has_include (<llama.h>)

#include "../LLamaDecisionContext.h"

class KevDecisionContext : public DecisionContext 
{
public:
    KevDecisionContext();

protected:

    llama_token token_marker = LLAMA_TOKEN_NULL;

    void InitSetup(const llama_model* model, const std::string& prefix) override;
   
    void FillModelTak(std::vector<llama_token>& tokens,
        const PreparedQuestion& question, DecisionPrompt& task) const override;

    void PreparePrompt(PromptInput& input) const override;

};

#endif
#endif
