#ifndef VTOL_TILTWING_LOG_HPP_
#define VTOL_TILTWING_LOG_HPP_

#include <iostream>
#include <string>
#include <ctime>
#include <map>
#include <fstream>
#include <chrono>

class Log {
public:
    // Severity levels based off of https://support.solarwinds.com/SuccessCenter/s/article/Syslog-Severity-levels.
    // See the article for an explanation of when to use which severity level.
    enum Severity {emergency = 0, alert, critical, error, warning, notice, info, debug};
    std::map<Severity, std::string> s_to_string;
    const unsigned int MAX_BUFFER_SIZE = 2048;
    unsigned int msg_len;
    bool print_and_save;
    std::string save_file;
protected:
    Severity severity;
    std::string datetime;
    std::string message;
    std::string log_string;
public:
    Log(std::string msg, Severity s);
    ~Log();
    static void write_to_file(const std::string& message, const std::string& filename);
    static void print(const std::string& message);
    template<typename T>
    std::string get_time(std::chrono::time_point<T> time);
};

#endif //VTOL_TILTWING_LOG_HPP_