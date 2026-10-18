#ifndef LLAMA_DECISION_LFM2_D1_H
#define LLAMA_DECISION_LFM2_D1_H

#if __has_include (<llama.h>)

#include "../LLamaDecisionContext.h"

class Lfm2D1DecisionContext : public DecisionContext 
{
public:
    Lfm2D1DecisionContext();

    bool SupportsMedia() const override;

protected:

    void InitSetup(const llama_model* model, const std::string& prefix) override;
    
    void PreparePrompt(PromptInput& input) const override;

    void FillModelTak(std::vector<llama_token>& tokens, const PreparedQuestion& question, DecisionPrompt& task) const override;
    bool BuildLabels(const Question& question, std::vector<std::string>& texts,
        std::vector<std::vector<llama_token>>& groups) const;
};

#endif
#endif
