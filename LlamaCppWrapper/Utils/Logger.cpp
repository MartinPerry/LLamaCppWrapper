#include "./Logger.h"

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <string.h>

#ifdef _WIN32
//#include <Windows.h>

//#include <DbgHelp.h>
//#pragma comment( lib, "dbghelp.lib" )

#	if defined(_MSC_VER) && defined(__clang__)
#		define WIN32_LEAN_AND_MEAN
#		define VC_EXTRALEAN
#	endif
#	include <windows.h>   // WinApi header
#endif

#ifdef __ANDROID_API__
#	include <android/log.h>
#endif

using namespace MyUtils;

std::shared_ptr<Logger> Logger::instanceLogger = nullptr;


Logger::Logger() :
	eol("\n"),
	colorsEnabled(false),
	enableErrors{ true, true, true },
	enableWarnings{ true, true, true },
#if defined(_DEBUG) || defined(DEBUG)
	enableInfo{ true, true, true },
#else
	enableInfo{ false, false, false },
#endif
	loggerOutput{ nullptr, nullptr, nullptr }
{

	this->LogToStdout();
}

Logger::~Logger()
{
	if (this->loggerOutput[LogOutput::File] != nullptr)
	{
		fclose(this->loggerOutput[LogOutput::File]);
		this->loggerOutput[LogOutput::File] = nullptr;
	}
}

void Logger::Initialize()
{
	instanceLogger = std::shared_ptr<Logger>(new Logger());
}

void Logger::Destroy()
{
	instanceLogger = nullptr;
}

std::shared_ptr<Logger> Logger::GetInstance()
{
	if (instanceLogger == nullptr)
	{
		Logger::Initialize();
	}

	return instanceLogger;
}

void Logger::LogToStdout()
{
	this->loggerOutput[LogOutput::StdOut] = stdout;
}

void Logger::LogToStder()
{
	this->loggerOutput[LogOutput::StdErr] = stderr;
}

void Logger::SetColorsEnabled(bool val)
{
	this->colorsEnabled = val;
}

void Logger::StartColor(int colorID)
{
	if (colorsEnabled == false)
	{
		return;
	}

#ifdef _WIN32
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, colorID);
#endif
}

void Logger::EndColor(int colorID)
{
	if (colorsEnabled == false)
	{
		return;
	}

#ifdef _WIN32
	HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
	SetConsoleTextAttribute(hConsole, colorID);
#endif
}

void Logger::SetLineEnd(const char* end)
{
	this->eol = end;
}

void Logger::LogError(const char* message, ...)
{
	auto l = Logger::GetInstance();



	va_list vl;
	va_start(vl, message);

	for (int i = 0; i < 3; i++)
	{
		if (l->loggerOutput[i] == nullptr)
		{
			continue;
		}
		if (l->enableErrors[i] == false)
		{
			continue;
		}

#ifdef __ANDROID_API__
		__android_log_vprint(ANDROID_LOG_ERROR, "CLogger", message, vl);
#else
		l->StartColor(Logger::COLOR_RED);
		fprintf(l->loggerOutput[i], "[Error] ");
		l->EndColor(Logger::COLOR_WHITE);

		vfprintf(l->loggerOutput[i], message, vl);
		fprintf(l->loggerOutput[i], l->eol.c_str());

		fflush(l->loggerOutput[i]);
#endif
	}
	va_end(vl);



}


void Logger::LogWarning(const char* message, ...)
{
	auto l = Logger::GetInstance();


	va_list vl;
	va_start(vl, message);

	for (int i = 0; i < 3; i++)
	{
		if (l->loggerOutput[i] == nullptr)
		{
			continue;
		}
		if (l->enableWarnings[i] == false)
		{
			continue;
		}

#ifdef __ANDROID_API__
		__android_log_vprint(ANDROID_LOG_WARN, "CLogger", message, vl);
#else
		l->StartColor(Logger::COLOR_YELLOW);
		fprintf(l->loggerOutput[i], "[Warning] ");
		l->EndColor(Logger::COLOR_WHITE);

		vfprintf(l->loggerOutput[i], message, vl);
		fprintf(l->loggerOutput[i], l->eol.c_str());
		fflush(l->loggerOutput[i]);
#endif
	}

	va_end(vl);


}

void Logger::LogInfo(const char* message, ...)
{
	auto l = Logger::GetInstance();



	va_list vl;
	va_start(vl, message);

	for (int i = 0; i < 3; i++)
	{
		if (l->loggerOutput[i] == nullptr)
		{
			continue;
		}
		if (l->enableInfo[i] == false)
		{
			continue;
		}

#ifdef __ANDROID_API__
		__android_log_vprint(ANDROID_LOG_INFO, "CLogger", message, vl);
#else
		l->StartColor(Logger::COLOR_GREEN);
		fprintf(l->loggerOutput[i], "[Info] ");
		l->EndColor(Logger::COLOR_WHITE);

		vfprintf(l->loggerOutput[i], message, vl);
		fprintf(l->loggerOutput[i], l->eol.c_str());
		fflush(l->loggerOutput[i]);
#endif

	}

	va_end(vl);


}

void Logger::LogMessage(const char* message, ...)
{
	auto l = Logger::GetInstance();

	va_list vl;
	va_start(vl, message);

	for (int i = 0; i < 3; i++)
	{
		if (l->loggerOutput[i] == nullptr)
		{
			continue;
		}

#ifdef __ANDROID_API__
		__android_log_vprint(ANDROID_LOG_DEBUG, "CLogger", message, vl);
#else
		vfprintf(l->loggerOutput[i], message, vl);
		fflush(l->loggerOutput[i]);
#endif
	}

	va_end(vl);


}

void Logger::Enable(LogType type, LogOutput output)
{
	switch (type)
	{
	case LogType::Error:
		this->enableErrors[output] = true;
		break;
	case LogType::Warning:
		this->enableWarnings[output] = true;
		break;
	case LogType::Info:
		this->enableInfo[output] = true;
		break;
	}
}

void Logger::Disable(LogType type, LogOutput output)
{
	switch (type)
	{
	case LogType::Error:
		this->enableErrors[output] = false;
		break;
	case LogType::Warning:
		this->enableWarnings[output] = false;
		break;
	case LogType::Info:
		this->enableInfo[output] = false;
		break;
	}
}

bool Logger::IsEnabled(LogType type, LogOutput output)
{
	switch (type)
	{
	case LogType::Error:
		return this->enableErrors[output];		
	case LogType::Warning:
		return this->enableWarnings[output];
	case LogType::Info:
		return this->enableInfo[output];
	}

	return false;
}
