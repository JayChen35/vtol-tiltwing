#include <iostream>
#include <string>
#include <ctime>
#pragma once

class Log() {
public:
    std::string datetime;
    std::string message;
    std::string log_string;
    // Severity levels based off of https://support.solarwinds.com/SuccessCenter/s/article/Syslog-Severity-levels.
    // See the article for an explanation of when to use which severity level.
    enum severity {debug, info, notice, warning, error, critical, alert, emergency};
    Log();
    ~Log();
    void execute();
    void write_to_file(std::string message);
    void print(std::string message);
}