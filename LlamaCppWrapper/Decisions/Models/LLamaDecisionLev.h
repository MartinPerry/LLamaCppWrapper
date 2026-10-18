#ifndef LLAMA_DECISION_LEV_H
#define LLAMA_DECISION_LEV_H

#if __has_include (<llama.h>)

#include "./LLamaDecisionCodeLabels.h"

class LevDecisionContext : public CodeLabelsDecisionContext
{
public:
    LevDecisionContext();

protected:
    
    size_t GetVariantsCount(const PreparedQuestion& question) const override;
    size_t GetOutputsCount(const PreparedQuestion& question) const override;
};

#endif
#endif
