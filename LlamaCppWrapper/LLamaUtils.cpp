#include "./LlamaUtils.h"

#if __has_include (<llama.h>)

#include <cmath>
#include <algorithm>
#include <iostream>

#include "./Utils/Logger.h"

std::string LLamaUtils::GetCmdInput(const std::string& userPrompt)
{
    std::cout << userPrompt << std::endl;
    std::string query;
    std::getline(std::cin, query);
    
    return query;
}

constexpr std::string_view LLamaUtils::trim(std::string_view text, std::string_view exclude)
{
    if (const size_t leftShift = text.find_first_not_of(exclude);
        leftShift != std::string_view::npos
        )
    {
        text.remove_prefix(leftShift);
    }
    else
    {
        return {};
    }

    if (const size_t rightShift = text.find_last_not_of(exclude);
        rightShift != std::string_view::npos
        )
    {
        text.remove_suffix(text.size() - rightShift - 1);
    }
    else
    {
        return {};
    }

    return text;
}

std::vector<std::string_view> LLamaUtils::SplitByLines(std::string_view utf8_data, size_t minLineLen)
{
    std::vector<std::string_view> lines;
    size_t start = 0;
    size_t end = 0;

    while ((end = utf8_data.find('\n', start)) != std::string::npos)
    {
        auto line = utf8_data.substr(start, end - start);
        line = LLamaUtils::trim(line);
        if (line.length() > minLineLen)
        {
            lines.push_back(line);
        }
        start = end + 1;
    }

    if (start < utf8_data.size())
    {
        // Add last line if no trailing newline
        auto line = utf8_data.substr(start);
        line = LLamaUtils::trim(line);
        if (line.length() > minLineLen)
        {
            lines.push_back(line);
        }
    }

    return lines;
}

std::vector<std::string_view> LLamaUtils::SplitByWords(std::string_view utf8_data)
{
    std::vector<std::string_view> words;
    size_t start = 0;
    size_t end = 0;

    while ((end = utf8_data.find_first_of(" \n", start)) != std::string::npos)
    {
        auto word = utf8_data.substr(start, end - start);
        word = LLamaUtils::trim(word);
        words.push_back(word);
        
        start = end + 1;
    }

    if (start < utf8_data.size())
    {
        // Add last word if no trailing space
        auto word = utf8_data.substr(start);
        word = LLamaUtils::trim(word);
        words.push_back(word);        
    }

    return words;
}

std::string LLamaUtils::MergeWords(const std::vector<std::string_view>& words, int startOffset, int wordsCount)
{
    std::string text = "";

    int startIndex = std::clamp<int>(startOffset, 0, words.size());
    int endIndex = std::min<int>(words.size(), startIndex + wordsCount);

    for (int i = startIndex; i < endIndex; i++)
    {
        text += words[i];
        text += " ";
    }

    return text;
}

#endif