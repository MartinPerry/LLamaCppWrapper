#include "./LLamaDecisionLfm2D1.h"

#if __has_include (<llama.h>)

#include "../../Utils/Logger.h"
#include <algorithm>
#include <cctype>

Lfm2D1DecisionContext::Lfm2D1DecisionContext() : 
    DecisionContextMultiModal(DecisionModelType::LFM2_D1)
{
}


void Lfm2D1DecisionContext::InitSetup(const llama_model*, const std::string&)
{    
    n_options_max = 255;
    noul_true_first = true;
    allow_null_state = true;
}

bool Lfm2D1DecisionContext::BuildLabels(const Question& question, std::vector<std::string>& texts, std::vector<std::vector<llama_token>>& groups) const
{
    const size_t n_options = question.criteria.size();

    auto get_single_tokens = [&](const std::vector<std::string> & forms) {
        std::vector<llama_token> out;
        for (const auto & form : forms) 
        {
            const auto toks = DecisionContext::tokenize(vocab, form, false, false);
            if (toks.size() == 1 && std::find(out.begin(), out.end(), toks[0]) == out.end()) 
            {
                out.push_back(toks[0]);
            }
        }

        return out;
    };

    if (question.type != QuestionType::CHOICE) 
    {
        for (const auto & opt : question.criteria) 
        {
            std::vector<llama_token> group;
            if (question.type == QuestionType::SCORE) 
            {
                group = get_single_tokens({opt.key});
            } 
            else if (opt.key == "true") 
            {
                group = get_single_tokens({"yes", "Yes", "YES"});
            } 
            else 
            {
                group = get_single_tokens({"no", "No", "NO"});
            }

            if (group.empty()) 
            {
                MY_LOG_ERROR("decision label is not a single token: %s", opt.key.c_str());
                return false;
            }

            texts.push_back(opt.key);
            groups.push_back(group);
        }
        return true;
    }

    bool is_letters = true;
    for (const auto & opt : question.criteria) 
    {
        is_letters = is_letters && opt.key.size() == 1 && std::isalpha((unsigned char) opt.key[0]);
    }

    std::vector<std::string> codes;
    for (size_t i = 0; i < n_options; i++) 
    {
        if (is_letters) 
        {
            codes.push_back(question.criteria[i].key);
        } 
        else if (n_options <= 26) 
        {
            codes.push_back(std::string(1, 'A' + i));
        } 
        else 
        {
            codes.push_back((i < 10 ? "0" : "") + std::to_string(i));
        }
    }

    std::vector<std::string> pool;
    for (char c = 'A'; c <= 'Z'; c++) 
    {
        pool.push_back(std::string(1, c));
    }

    for (int i = 0; i < 100; i++) 
    {
        pool.push_back((i < 10 ? "0" : "") + std::to_string(i));
    }

    for (char c = 'a'; c <= 'z'; c++) 
    {
        pool.push_back(std::string(1, c));
    }

    for (int i = 0; i < 200; i++) 
    {
        pool.push_back("#" + std::to_string(i));
    }

    for (char a = 'A'; a <= 'Z'; a++) 
    {
        for (char b = 'A'; b <= 'Z'; b++) 
        {
            pool.push_back(std::string{a, b});
        }
    }

    std::vector<llama_token> used;
    auto take = [&](const std::string & code) {

        const auto toks = DecisionContext::tokenize(vocab, code, false, false);
        if (toks.size() != 1 || std::find(used.begin(), used.end(), toks[0]) != used.end()) 
        {
            return false;
        }

        used.push_back(toks[0]);
        std::vector<llama_token> group = {toks[0]};
        for (const llama_token tok : get_single_tokens({" " + code})) 
        {
            if (tok != toks[0]) 
            {
                group.push_back(tok);
            }
        }

        texts.push_back(code);
        groups.push_back(group);
        return true;
    };

    for (const auto & code : codes) 
    {
        bool is_taken = take(code);
        for (size_t i = 0; !is_taken && i < pool.size(); i++) 
        {
            is_taken = take(pool[i]);
        }

        if (!is_taken)
        {
            MY_LOG_ERROR("no single-token label left for %zu options", n_options);
            return false;
        }
    }

    return true;
}


void Lfm2D1DecisionContext::PreparePrompt(PromptInput& input) const
{
    std::vector<std::vector<llama_token>> groups;
    if (!this->BuildLabels(input.question, input.labels, groups))
    {
        input.labels.clear();
    }
}

void Lfm2D1DecisionContext::FillModelTak(std::vector<llama_token>&, const PreparedQuestion& question, DecisionPrompt& task) const
{
    std::vector<std::string> texts;
    std::vector<std::vector<llama_token>> groups;
    if (!this->BuildLabels(question, texts, groups))
    {
        return;
    }

    for (const auto& group : groups) 
    {
        task.labels.insert(task.labels.end(), group.begin(), group.end());
        task.label_groups.push_back(group.size());
    }
}

#endif
