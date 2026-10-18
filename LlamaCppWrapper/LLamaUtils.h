#ifndef LLAMA_UTILS_H
#define LLAMA_UTILS_H

#if __has_include (<llama.h>)

#include <vector>
#include <list>
#include <string>
#include <limits>

class LLamaUtils 
{
public:

	static std::string GetCmdInput(const std::string& userPrompt);

	static std::vector<std::string_view> SplitByLines(std::string_view utf8_data, size_t minLineLen);
	static std::vector<std::string_view> SplitByWords(std::string_view utf8_data);

	static std::string MergeWords(const std::vector<std::string_view>& words, 
		int startOffset = 0, 
		int wordsCount = std::numeric_limits<int>::max());

protected:
	static constexpr std::string_view trim(std::string_view text, std::string_view exclude = " \n\r\t\v\0");

};

#endif
#endif