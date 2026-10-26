#ifndef LLAMA_DECISION_LFM2_D1_OMNI_H
#define LLAMA_DECISION_LFM2_D1_OMNI_H

#if __has_include (<llama.h>)

#include "../LLamaDecisionContextMultiModal.h"

class Lfm2D1OmniDecisionContext : public DecisionContextMultiModal
{
public:
    Lfm2D1OmniDecisionContext();

    void FillTask(const std::string& state, const std::vector<PreparedQuestion>& questions,
        const PreparedQuestion& question, size_t variant, DecisionPrompt& task) const override;

protected:

    llama_token token_marker = LLAMA_TOKEN_NULL;

    void InitSetup(const llama_model* model, const std::string& prefix) override;
    void PreparePrompt(PromptInput& input) const override;
};

#endif
#endif
