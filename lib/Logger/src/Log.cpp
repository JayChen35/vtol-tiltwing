#include <iostream>
#include <string>
#include <ctime>
#include <format>
#pragma once

class Log() {
public:
    // Severity levels based off of https://support.solarwinds.com/SuccessCenter/s/article/Syslog-Severity-levels.
    // See the article for an explanation of when to use which severity level.
    enum Severity {emergency = 0, alert, critical, error, warning, notice, info, debug};
protected:
    Severity severity;
    std::string datetime;
    std::string message;
    std::string log_string;
public:
    Log(std::string msg, Severity s) {
        message = msg;
        severity = s;
        time_t now = time(0);
        char* datetime = ctime(&now); // Convert now to string form
        switch (s) {
            case emergency: {log_string = std::format("[{}][{}][{}]: {}", severity, "EMERGENCY", datetime, message)}
            case alert: {log_string = std::format("[{}][{}][{}]: {}", severity, "ALERT", datetime, message)}
            case critical: {log_string = std::format("[{}][{}][{}]: {}", severity, "CRITICAL", datetime, message)}
            case error: {log_string = std::format("[{}][{}][{}]: {}", severity, "ERROR", datetime, message)}
            case warning: {log_string = std::format("[{}][{}][{}]: {}", severity, "WARNING", datetime, message)}
            case notice: {log_string = std::format("[{}][{}][{}]: {}", severity, "NOTICE", datetime, message)}
            case info: {log_string = std::format("[{}][{}][{}]: {}", severity, "INFO", datetime, message)}
            case debug: {log_string = std::format("[{}][{}][{}]: {}", severity, "DEBUG", datetime, message)}
        }
        execute();
    }
    ~Log();
    void execute() {
    }
    void write_to_file(std::string message);
    void print(std::string message);
}