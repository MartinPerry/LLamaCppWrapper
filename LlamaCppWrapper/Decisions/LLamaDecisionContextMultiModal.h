#ifndef LLAMA_DECISION_CONTEXT_MULTI_MODAL_H
#define LLAMA_DECISION_CONTEXT_MULTI_MODAL_H

#if __has_include(<llama.h>) && __has_include(<mtmd.h>)

class DecisionTemplate;

#include <map>
#include <unordered_map>
#include <vector>

#include <mtmd.h>

#include "./LLamaDecisionStructures.h"
#include "./LLamaDecision.h"
#include "./LLamaDecisionContext.h"
#include "./LLamaDecisionMultiModal.h"

struct DecisionContextMultiModal : public DecisionContext
{		
public:
	static bool Tokenize(mtmd_context* ctx_vision,
		const std::string& prompt,
		const std::vector<mtmd_bitmap*>& bitmaps,
		DecisionPrompt& task);

	virtual ~DecisionContextMultiModal() = default;
	
	friend class LLamaDecisionMultiModal;

protected:

	// Some models (Clef, Lfm2D1Omni) require access to input data
	// So we pass it via dependecy injection 
	// Valid only during RunInternal
	const DecisionMediaInput* mediaInput = nullptr;

	DecisionContextMultiModal(DecisionModelType modelType);

	void Init(const llama_model* model, const std::string& prefix) override;
	
	std::string RunTemplate(const PromptInput& input) const override;
	bool TokenizePrompt(const std::string& prompt, DecisionPrompt& task) const override;

	float GetTemperature(const PreparedQuestion& question) const override;

};


#endif
#endif
