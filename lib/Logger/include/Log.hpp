#include <iostream>
#include <string>
#include <ctime>
#include <format>
#pragma once

class Log() {
public:
    // Severity levels based off of https://support.solarwinds.com/SuccessCenter/s/article/Syslog-Severity-levels.
    // See the article for an explanation of when to use which severity level.
    enum Severity {emergency, alert, critical, error, warning, notice, info, debug};
protected:
    Severity severity;
    std::string datetime;
    std::string message;
    std::string log_string;
public:
    Log(std::string msg, Severity s);
    ~Log();
    void execute();
    void write_to_file(std::string message);
    void print(std::string message);
}