#include "./LLamaDecisionLfm2D1Omni.h"

#if __has_include (<llama.h>)

#include "../../Utils/Logger.h"

#include <algorithm>

static const std::string D1OMNI_MARKER = "<<d1omni:";
static const std::string D1OMNI_SEP = "<<d1omni:sep>>";
static const std::string D1OMNI_MARK_STATE = "<<d1omni:state>>";
static const std::string D1OMNI_MARK_QUESTION = "<<d1omni:question>>";
static const std::string D1OMNI_MARK_OPTION = "<<d1omni:option>>";

Lfm2D1OmniDecisionContext::Lfm2D1OmniDecisionContext() : 
    DecisionContext(DecisionModelType::LFM2_D1_OMNI) 
{
}

bool Lfm2D1OmniDecisionContext::SupportsMedia() const
{
    return true;
}

void Lfm2D1OmniDecisionContext::InitSetup(const llama_model*, const std::string&)
{    
    token_marker = llama_vocab_mask(vocab);
    if (token_marker == LLAMA_TOKEN_NULL) {
        MY_LOG_ERROR("decision model has no mask token");
        return;
    }
    n_options_max = 255;
    allow_noul_aliases = true;
    allow_null_state = true;  
}

void Lfm2D1OmniDecisionContext::PreparePrompt(PromptInput& input) const
{
    auto clean = [](std::string& text) {
        text = DecisionContext::escape_special_tokens(text);
        DecisionContext::replace_all(text, D1OMNI_MARKER, "<<d1omni ");
    };

    clean(input.state);
    clean(input.question.id);
    clean(input.question.instructions);
    for (auto& option : input.question.criteria) 
    {
        clean(option.key);
        if (option.value)
        {
            clean(*option.value);
        }
    }

    input.separator = D1OMNI_SEP;
    input.stateMarker = D1OMNI_MARK_STATE;
    input.questionMarker = D1OMNI_MARK_QUESTION;
    input.optionMarker = D1OMNI_MARK_OPTION;
}

void Lfm2D1OmniDecisionContext::FillTask(const std::string& state, const std::vector<PreparedQuestion>& questions,
    const PreparedQuestion& question, size_t variant, DecisionPrompt& task) const
{
    task = {};
    std::string mediaMarkers;
    if (mediaInput) {
        if (mediaInput->audio && mediaInput->markers.size() != 1) {
            MY_LOG_ERROR("a request has images or one audio clip, not both");
            return;
        }
        for (const auto& marker : mediaInput->markers) mediaMarkers += marker;
        if (!this->TokenizePrompt(mediaMarkers, task)) {
            task = {};
            return;
        }
    }
    const size_t n_media = task.tokens.size();
    const std::string prompt = this->Render(state, questions, question, variant);
    const auto pieces = DecisionContext::split(prompt, D1OMNI_SEP);
    if (pieces.empty() || pieces[0] != mediaMarkers || question.criteria.empty()) 
    {
        MY_LOG_ERROR("unexpected layout of the decision prompt");
        task = {};
        return;
    }
    size_t n_max = 16384;
    if (mediaInput) {
        n_max = std::min<size_t>(mediaInput->audio ? 15360 : 896, 16384 - std::min<size_t>(16384, n_media));
        if (n_max < 64) {
            MY_LOG_ERROR("the media leave too few positions for the decision, send fewer images");
            task = {};
            return;
        }
    }
    // the options get max(96, min(24 n + 32, max / 2)) tokens, shared evenly
    const int64_t n_options      = question.criteria.size();
    const int64_t n_budget       = std::max<int64_t>(96, std::min<int64_t>(24 * n_options + 32, n_max / 2));
    const int64_t n_option_max   = std::max<int64_t>(2, (n_budget - 3 * n_options) / n_options);
    const int64_t n_question_max = std::max<int64_t>(16, n_budget);

    std::vector<llama_token> head;  // before the state
    std::vector<llama_token> body;  // the state
    std::vector<llama_token> tail;  // after the state
    bool has_state = false;
    for (size_t i_piece = 1; i_piece < pieces.size(); i_piece++) {
        std::string piece = pieces[i_piece];
        int64_t n_piece_max = -1;
        bool is_state = false;
        if (DecisionContext::starts_with(piece, D1OMNI_MARK_STATE)) {
            piece    = piece.substr(D1OMNI_MARK_STATE.size());
            is_state = true;
        } else if (DecisionContext::starts_with(piece, D1OMNI_MARK_QUESTION)) {
            piece = piece.substr(D1OMNI_MARK_QUESTION.size());
            n_piece_max = n_question_max;
        } else if (DecisionContext::starts_with(piece, D1OMNI_MARK_OPTION)) {
            piece = piece.substr(D1OMNI_MARK_OPTION.size());
            n_piece_max = n_option_max;
        }

        std::vector<llama_token> tokens = DecisionContext::tokenize(vocab, piece, false, true);
        if (n_piece_max >= 0 && (int64_t) tokens.size() > n_piece_max) {
            tokens.resize(n_piece_max);
        }

        if (is_state) {
            if (has_state) {
                MY_LOG_ERROR("unexpected layout of the decision prompt");
                task = {};
                return;
            }
            body      = std::move(tokens);
            has_state = true;
        } else {
            std::vector<llama_token> & dst = has_state ? tail : head;
            dst.insert(dst.end(), tokens.begin(), tokens.end());
        }
    }
    if (!has_state) {
        MY_LOG_ERROR("unexpected layout of the decision prompt");
        task = {};
        return;
    }

    const size_t n_room = n_max - std::min(n_max, head.size() + tail.size());
    body.resize(std::min(body.size(), n_room));

    std::vector<llama_token> tokens = std::move(head);
    tokens.insert(tokens.end(), body.begin(), body.end());
    tokens.insert(tokens.end(), tail.begin(), tail.end());
    tokens.resize(std::min(tokens.size(), n_max));

    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == token_marker) {
            task.markers.push_back(static_cast<int32_t>(n_media + i));
        }
    }
    if ((int64_t) task.markers.size() != n_options) {
        MY_LOG_ERROR("the options do not fit in the context");
        task = {};
        return;
    }

    // the output has one score per question type
    task.column = static_cast<int32_t>(question.type);
    task.tokens.insert(task.tokens.end(), tokens.begin(), tokens.end());
}

#endif
