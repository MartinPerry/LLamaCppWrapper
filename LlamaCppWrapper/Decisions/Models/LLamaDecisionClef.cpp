#include "./LLamaDecisionClef.h"

#if __has_include (<llama.h>)

#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include "../../Utils/Logger.h"

// Adapted from the supplied server-decision.cpp/.h and the decision output
// extraction in server-context.cpp. No server, HTTP, task queue or mtmd linkage.
#include <llama-ext.h>
#include <llama.h>

ClefDecisionContext::ClefDecisionContext(DecisionModelType type) :
	DecisionContextMultiModal(type)
{
}


bool ClefDecisionContext::IsJointTask() const
{
	return true;
}

void ClefDecisionContext::InitSetup(const llama_model* model, const std::string& prefix)
{	
	n_options_max = 255;
	noul_true_first = true;		
}



//
// joint prompt (clef)
//

// given to the template: text between the pieces of the prompt, and at the start of the span of a question or of an option
static const std::string CLEF_MARKER = "<<clef:";
static const std::string CLEF_SEP = "<<clef:sep>>";
static const std::string CLEF_MARK_QUESTION = "<<clef:question>>";
static const std::string CLEF_MARK_OPTION = "<<clef:option>>";

void ClefDecisionContext::FillTask(const std::string& state, 
	const std::vector<PreparedQuestion>& questions,
	const PreparedQuestion& question, 
	size_t variant, 
	DecisionPrompt& task) const 
{	
	PromptInput input;
	input.state = state;
	input.joint = true;
	input.modelType = modelType;
	input.separator = CLEF_SEP;
	input.questionMarker = CLEF_MARK_QUESTION;
	input.optionMarker = CLEF_MARK_OPTION;
	
	DecisionContext::replace_all(input.state, CLEF_MARKER, "<<clef ");
	for (const auto& q : questions) 
	{
		Question rendered = q;
		DecisionContext::replace_all(rendered.id, CLEF_MARKER, "<<clef ");
		DecisionContext::replace_all(rendered.instructions, CLEF_MARKER, "<<clef ");

		for (auto& option : rendered.criteria) 
		{
			DecisionContext::replace_all(option.key, CLEF_MARKER, "<<clef ");

			if (option.value)
			{
				DecisionContext::replace_all(*option.value, CLEF_MARKER, "<<clef ");
			}
		}

		input.questions.push_back(std::move(rendered));
	}

	const std::string prompt = this->RunTemplate(input);

	const auto invalid = std::runtime_error("unexpected layout of the decision prompt");

	// Keep media in the prefix and tokenize the decision spans one piece at a time.
	const std::vector<std::string> pieces = DecisionContext::split(prompt, CLEF_SEP);	
	task = {}; // Rebuilding a task must not retain its previous output layout.
	size_t i_question = 0;
    size_t firstPiece = 0;
    
	if (mediaInput) 
	{
        std::string head;
        const std::string& marker = mediaInput->markers.front();
        while (firstPiece < pieces.size() && head.find(marker) == std::string::npos) 
		{
            head += pieces[firstPiece++];
        }

        if (head.find(marker) == std::string::npos || head.find(CLEF_MARKER) != std::string::npos ||
            !this->TokenizePrompt(head, task)) 
		{
            MY_LOG_ERROR("unexpected layout of the multimodal decision prompt");
            task = {};
            return;
        }

        task.order.resize(task.tokens.size(), LLAMA_DECISION_ORDER_NONE);
    }

	for (size_t i_piece = firstPiece; i_piece < pieces.size(); ++i_piece) 
	{
		std::string piece = pieces[i_piece];
		int32_t order = LLAMA_DECISION_ORDER_NONE;

		if (DecisionContext::starts_with(piece, CLEF_MARK_QUESTION)) 
		{
			piece = piece.substr(CLEF_MARK_QUESTION.size());
			if (i_question >= questions.size()) 
			{
				throw invalid;
			}
			switch (questions[i_question++].type) 
			{
				case QuestionType::NOUL:   
					order = LLAMA_DECISION_ORDER_QUESTION_NOUL;   
					break;
				case QuestionType::CHOICE: 
					order = LLAMA_DECISION_ORDER_QUESTION_CHOICE; 
					break;
				case QuestionType::SCORE:  
					order = LLAMA_DECISION_ORDER_QUESTION_SCORE;  
					break;
			}
		}
		else if (DecisionContext::starts_with(piece, CLEF_MARK_OPTION)) 
		{
			piece = piece.substr(CLEF_MARK_OPTION.size());
			order = LLAMA_DECISION_ORDER_OPTION;
			task.n_scores++;
		}

		const std::vector<llama_token> piece_tokens = DecisionContext::tokenize(vocab, piece, false, true);
		if (order != LLAMA_DECISION_ORDER_NONE && piece_tokens.empty()) 
		{
			throw std::invalid_argument("the instructions and the options of a question must not be empty");
		}

		task.tokens.insert(task.tokens.end(), piece_tokens.begin(), piece_tokens.end());
		task.order.resize(task.tokens.size(), order);
	}

	size_t n_options = 0;
	for (const auto& question : questions) 
	{
		n_options += question.criteria.size();
	}

	if (i_question != questions.size() || (size_t)task.n_scores != n_options) 
	{
		throw invalid;
	}

}


#endif
