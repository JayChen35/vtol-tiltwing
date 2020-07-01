#ifndef VTOL_TILTWING_LOGGER_HPP_
#define VTOL_TILTWING_LOGGER_HPP_

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

#endif // VTOL_TILTWING_LOGGER_HPP_