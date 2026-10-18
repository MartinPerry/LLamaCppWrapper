#ifndef LOGGER_H
#define LOGGER_H


#ifndef MY_LOG_ERROR
#   define MY_LOG_ERROR(...) MyUtils::Logger::LogError(__VA_ARGS__)
#endif

#ifndef MY_LOG_WARNING
#   define MY_LOG_WARNING(...) MyUtils::Logger::LogWarning(__VA_ARGS__)
#endif

#ifndef MY_LOG_INFO
#	if defined(_DEBUG) || defined(DEBUG) || defined (USE_RELEASE_LOG_INFO)
#		define MY_LOG_INFO(...) MyUtils::Logger::LogInfo(__VA_ARGS__)
#	else
#		define MY_LOG_INFO(...)
#	endif
#endif

#ifndef MY_LOG
#   define MY_LOG(...) MyUtils::Logger::LogMessage(__VA_ARGS__)
#endif

#include <cstdio>
#include <memory>
#include <string>

namespace MyUtils
{
	class Logger
	{
	public:
		enum LogOutput { StdOut = 0, File = 1, StdErr = 2 };
		enum LogType { Error = 0, Warning = 1, Info = 2 };

		static void Initialize();
		static void Destroy();
		static std::shared_ptr<Logger> GetInstance();

		static void LogError(const char* message, ...);
		static void LogWarning(const char* message, ...);
		static void LogInfo(const char* message, ...);
		static void LogMessage(const char* message, ...);

		~Logger();

		void SetColorsEnabled(bool val);
		
		void LogToStdout();
		void LogToStder();

		void SetLineEnd(const char* end);

		void Enable(LogType type, LogOutput output);
		void Disable(LogType type, LogOutput output);
		bool IsEnabled(LogType type, LogOutput output);
		
	private:
#ifdef _WIN32
		static const int COLOR_RED = 12;
		static const int COLOR_GREEN = 10;
		static const int COLOR_YELLOW = 14;
		static const int COLOR_WHITE = 15;
		static const int COLOR_BLUE = 9;
		static const int COLOR_DARK_GRAY = 8;
#else
		static const int COLOR_RED = 12;
		static const int COLOR_GREEN = 10;
		static const int COLOR_YELLOW = 14;
		static const int COLOR_WHITE = 15;
		static const int COLOR_BLUE = 9;
		static const int COLOR_DARK_GRAY = 8;
#endif


		static std::shared_ptr<Logger> instanceLogger;

		Logger();

		std::string eol;

		bool colorsEnabled;

		bool enableErrors[3];
		bool enableWarnings[3];
		bool enableInfo[3];

		FILE* loggerOutput[3];
		

		void StartColor(int colorID);
		void EndColor(int colorID);


	};
}


#endif
