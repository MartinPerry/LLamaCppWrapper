#ifndef LLAMA_DECISION_LAYA_H
#define LLAMA_DECISION_LAYA_H

#if __has_include (<llama.h>)

#include <string>

#include "../LLamaDecisionContext.h"

class LayaDecisionContext : public DecisionContext 
{
public:
    LayaDecisionContext();

protected:

    std::string text_marker;
    llama_token token_marker = LLAMA_TOKEN_NULL;
    llama_token token_sep = LLAMA_TOKEN_NULL;

    size_t max_head_tokens = 0;
    size_t max_option_tokens = 48;

    void InitSetup(const llama_model* model, const std::string& prefix) override;
   
    void FillModelTak(std::vector<llama_token>& tokens,
        const PreparedQuestion& question, DecisionPrompt& task) const override;

    void PreparePrompt(PromptInput& input) const override;

};

#endif
#endif
