#ifndef LLAMA_DECISION_OPENJEV_H
#define LLAMA_DECISION_OPENJEV_H

#if __has_include (<llama.h>)

#include "../LLamaDecisionContext.h"

class OpenJevDecisionContext : public DecisionContext 
{
public:
    OpenJevDecisionContext();

    bool SupportsMedia() const override;

protected:

    void InitSetup(const llama_model* model, const std::string& prefix) override;
   
    void FillModelTak(std::vector<llama_token>& tokens,
        const PreparedQuestion& question, DecisionPrompt& task) const override;
};

#endif
#endif
