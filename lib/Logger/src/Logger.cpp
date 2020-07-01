#include <string>
#include <Log.hpp>
#include "Logger.hpp"

Logger::Logger() = default;
Logger::~Logger() = default;
void Logger::log(const std::string& msg, Log::Severity s) {
    Log temp_log = Log(msg, s);
    log_history.push_back(temp_log);
}
