#ifndef LLAMA_DECISION_H
#define LLAMA_DECISION_H

#if __has_include (<llama.h>)

struct DecisionContext;

#include <memory>
#include <functional>
#include <string>
#include <string_view>
#include <optional>
#include <unordered_map>
#include <vector>

#include <llama.h>
#include <llama-cpp.h>

#include "../LLamaWrapper.h"

#include "./LLamaDecisionStructures.h"

// Stateless, text-only TypeSafe decisions. One instance owns one model/context;
// calls on the same instance must be serialized by the caller.
class LLamaDecision : public LLamaWrapper
{
public:    

    // Loads the model immediately, like LLamaEmbedding.
    LLamaDecision(const std::string& modelPath, const Settings& sets);
    ~LLamaDecision() override;

    LLamaDecision(const LLamaDecision&) = delete;
    LLamaDecision& operator=(const LLamaDecision&) = delete;
    LLamaDecision(LLamaDecision&&) = delete;
    LLamaDecision& operator=(LLamaDecision&&) = delete;

    void InitModel() override;
    

    // Overrides the default systemone Jinja builder. An empty callback restores
    // the default. Receives normalized, model-specific template inputs.
    void SetCustomPromptBuilder(CustomPromptBuilder callback);

    
    std::unordered_map<std::string, DecisionAnswer> Run(const std::string& state, const std::vector<Question>& questions);
    std::unordered_map<std::string, DecisionAnswer> Run(const std::vector<Question>& questions);

protected:
    std::unique_ptr<DecisionContext> dcx;

    std::unordered_map<std::string, DecisionAnswer> RunInternal(
        const std::string& state,
        const std::vector<Question>& questions
    );

    // May remap the output layout to the final batch after prefilling media.
    virtual bool ProcessPrompt(DecisionPrompt& prompt);

private:
    static Settings PrepareSettings(const Settings& sets);
    std::vector<float> Evaluate(llama_context* ctx, const llama_model* model,
        const DecisionPrompt& prompt);
    std::vector<float> ReadScores(llama_context* ctx, const llama_model* model,
        const DecisionPrompt& prompt);
};

#endif
#endif
