#ifndef LLAMA_DECISION_STRUCTURES_H
#define LLAMA_DECISION_STRUCTURES_H

#if __has_include (<llama.h>)

#include <string>
#include <optional>
#include <vector>
#include <functional>
#include <memory>

#include <llama.h>

// typed decision models, see "<arch>.decision.type" in the model metadata
enum class DecisionModelType
{
    UNKNOWN, // a decision model of a type that is not supported

    OPENJEV, // logits of one label token per option, read at the last prompt token
    LEV,     // same as openjev, noul is read from a rating scale
    KEV,     // dot product of the hidden states of the last token and of one end token per option
    NIMBLE,  // same as openjev, the prompt lists all the questions of the request
    LAYA,    // score of one marker token per option, read from the embeddings output
    CLEF,    // all questions in one prompt, score of option i read from the embeddings output at row i    
    GRANITE_SWITCH,
    PPLX_DECIDER,
    VLLM_DECISION2,
    VLLM_D3,
    LFM2_D1,
    LFM2_D1_OMNI,
};


enum class QuestionType
{
    CHOICE, //options, with optional descriptions => the top option, plus a probability per option
    SCORE,  //2 to 10 levels, lowest first  => the expected level (can fall between two)
    NOUL    //a yes / no question 	=> the probability of yes
};

static inline const char* QuestionTypeToString(QuestionType type)
{
    switch (type)
    {
        case QuestionType::CHOICE: return "choice";
        case QuestionType::SCORE: return "score";
        case QuestionType::NOUL: return "noul";
    }
    return "";
}

// CHOICE: key identifies the option; value optionally describes it.
// SCORE: value, or key when value is absent, describes an ordered level.
// NOUL: optional criteria use the keys "false" and "true".
struct Option
{
    std::string key;
    std::optional<std::string> value = std::nullopt;
};

struct Question
{
    std::string id;
    QuestionType type = QuestionType::CHOICE;
    std::string instructions;
    std::vector<Option> criteria;
    
    // VLLM-D3 may use the model template's default instructions.
    bool useDefaultInstructions = false;
};

struct PreparedQuestion : Question
{
    // Model-order option -> original criteria index. NOUL may synthesize options.
    std::vector<size_t> originalIndices;
    //size_t answerSize = 0;
};

//===================================================================

struct DecisionAnswer
{
    std::string id;
    float confidence = 0.0f; // Choice/score confidence; NOUL value from 0 to 1.
    // Same order as the caller's criteria. Empty NOUL criteria use [false, true].
    // For LEV NOUL, these are [1 - rating, rating], mapped to the criteria order.
    std::vector<float> probabs;
};

struct mtmd_input_chunk;

struct DecisionMediaChunk
{
    size_t offset = 0; // Index in DecisionPrompt::tokens; media rows use LLAMA_TOKEN_NULL.
    std::shared_ptr<const mtmd_input_chunk> chunk;
};

struct DecisionPrompt
{
    std::vector<llama_token> tokens;

    std::vector<llama_token> labels;
    std::vector<int32_t> markers;
    int32_t column = 0;
    int32_t pointer = -1;
    std::vector<int32_t> order;
    int32_t n_scores = 0;
    
    // Number of alternative label tokens for each option (LFM2-D1).
    std::vector<size_t> label_groups;
    std::vector<DecisionMediaChunk> mediaChunks;
};

struct PromptInput
{
    std::string state;
    Question question; // Current question; unused for joint prompts.
    std::vector<Question> questions; // All questions for NIMBLE/CLEF.
    // Option labels for LEV/NIMBLE; LEV NOUL instead has nine rating labels.
    std::vector<std::string> labels;
    std::vector<std::vector<std::string>> questionLabels;
    DecisionModelType modelType;
    size_t variant = 0;
    bool joint = false;
    std::string separator;
    std::string questionMarker;
    std::string optionMarker;
    std::string stateMarker;
    bool audio = false;
    bool stateIsNull = false;
    std::vector<std::string> images; // One MTMD marker per image/audio input, as in server-decision.
};

using CustomPromptBuilder = std::function<std::string(const PromptInput&)>;

// Internal per-call bridge. Text-only users do not need to include or link MTMD.
struct DecisionMediaInput
{
    std::vector<std::string> markers;
    bool audio = false;
    std::function<bool(const std::string&, DecisionPrompt&)> tokenize;
};

#endif
#endif
