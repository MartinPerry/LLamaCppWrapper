#ifndef LLAMA_DECISION_CLEF_H
#define LLAMA_DECISION_CLEF_H

#if __has_include (<llama.h>)

#include "../LLamaDecisionContextMultiModal.h"

class ClefDecisionContext : public DecisionContextMultiModal
{
public:
    ClefDecisionContext(DecisionModelType type = DecisionModelType::CLEF);
    
    bool IsJointTask() const override;

    void FillTask(const std::string& state, const std::vector<PreparedQuestion>& questions,
        const PreparedQuestion& question, size_t variant, DecisionPrompt& task) const override;

protected:

    void InitSetup(const llama_model* model, const std::string& prefix) override;
    
};

#endif
#endif
