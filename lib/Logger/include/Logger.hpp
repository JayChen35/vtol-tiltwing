#ifndef VTOL_TILTWING_LOGGER_HPP
#define VTOL_TILTWING_LOGGER_HPP

#include <iostream>
#include <string>
#include <vector>
#include <Log.hpp>

class Logger {
public:
    std::vector<Log> log_history;
    Logger();
    ~Logger();
    void log(const std::string& msg, Log::Severity s);
};

#endif // VTOL_TILTWING_LOGGER_HPP
