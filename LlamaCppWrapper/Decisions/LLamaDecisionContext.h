#ifndef LLAMA_DECISION_CONTEXT_H
#define LLAMA_DECISION_CONTEXT_H

#if __has_include (<llama.h>)

class DecisionTemplate;

#include <map>
#include <unordered_map>

#include "./LLamaDecision.h"



struct DecisionContext 
{
	static bool starts_with(const std::string& value, const std::string& prefix);
	static void replace_all(std::string& value, const std::string& search, const std::string& replacement);
	static std::vector<std::string> split(const std::string& value, const std::string& separator);
	static std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text,
		bool addSpecial, bool parseSpecial);
	static std::string token_to_piece(const llama_vocab* vocab, llama_token token);

	static std::string meta_str(const llama_model* model, const std::string& key);
	static std::string escape_special_tokens(const std::string& text);
	static std::vector<float> AggregateLabelScores(const std::vector<float>& scores,
		const std::vector<size_t>& groups);
	

	static std::unique_ptr<DecisionContext> Create(const llama_model* model);

	virtual ~DecisionContext() = default;
		
	CustomPromptBuilder customPromptBuilder;
	std::shared_ptr<DecisionTemplate> tpl = nullptr;
			
	std::vector<PreparedQuestion> PrepareQuestions(const std::vector<Question>& input) const;
	
	virtual size_t GetVariantsCount(const PreparedQuestion& question) const;

	virtual bool IsJointTask() const;
	virtual void FillTask(const std::string& state, const std::vector<PreparedQuestion>& questions,
		const PreparedQuestion& question, size_t variant, DecisionPrompt& task) const;
	
	DecisionAnswer FormatAnswer(const PreparedQuestion& question, const std::vector<std::vector<float>>& scores) const;
	bool SetNullState(bool value);
	    
protected:
	const DecisionModelType modelType;

	const llama_vocab* vocab = nullptr;

	std::unordered_map<std::string, float> temperatures;
	std::map<size_t, std::vector<float>> score_bias;
	size_t n_options_max = 0;
	bool noul_true_first = false;
	
	bool noul_criteria_order = false;
	bool strict_payload = false;
	bool allow_noul_aliases = false;
	bool allow_null_state = false;
	bool state_is_null = false;
	bool entropy_confidence = false;

	std::vector<llama_token> labels;
	std::vector<std::string> label_texts;
				
	DecisionContext(DecisionModelType modelType);
	
	virtual void Init(const llama_model* model, const std::string& prefix);
	
	
	virtual std::string RunTemplate(const PromptInput& input) const;
    virtual bool TokenizePrompt(const std::string& prompt, DecisionPrompt& task) const;

	std::string Render(const std::string& state, const std::vector<PreparedQuestion>& questions,
		const PreparedQuestion& question, size_t variant) const;
	
	virtual size_t GetOutputsCount(const PreparedQuestion& question) const;
	
	virtual float GetTemperature(const PreparedQuestion& question) const;


	//model-specific overloads
	virtual void InitSetup(const llama_model* model, const std::string& prefix) {};
	virtual void FillModelTak(std::vector<llama_token>& tokens, 
		const PreparedQuestion& question, DecisionPrompt& task) const {}
	virtual void PreparePrompt(PromptInput& input) const {}
	

};


#endif
#endif
