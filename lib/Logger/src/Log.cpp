#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include "Log.hpp"


Log::Log(std::string msg, Severity s) {
    print_and_save = true;
    save_file = "log_events.txt";
    message = std::move(msg); // Using std::move() to avoid unnecessary copies
    // Expand the buffer if the message is greater than the max buffer size
    msg_len = message.length();
    if (msg_len > MAX_BUFFER_SIZE) {
        std::cout << "Message: " << message.substr(0, 10) << " [...] " << message.substr(msg_len-11, 10) << std::endl;
        std::cout << "Message exceeds the maximum char buffer size for Logs, aborting save/print." << std::endl;
        print_and_save = false;
    }
    severity = s;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    datetime = get_time(now);
    // TODO: Get rid of the following lines and make s_to_string a const static map
    s_to_string.insert(std::pair<Severity, std::string>(emergency, "EMERGENCY"));
    s_to_string.insert(std::pair<Severity, std::string>(alert, "ALERT"));
    s_to_string.insert(std::pair<Severity, std::string>(critical, "CRITICAL"));
    s_to_string.insert(std::pair<Severity, std::string>(error, "ERROR"));
    s_to_string.insert(std::pair<Severity, std::string>(warning, "WARNING"));
    s_to_string.insert(std::pair<Severity, std::string>(notice, "NOTICE"));
    s_to_string.insert(std::pair<Severity, std::string>(info, "INFO"));
    s_to_string.insert(std::pair<Severity, std::string>(debug, "DEBUG"));
    log_string = "[" + std::to_string(static_cast<int>(severity)) + "]" + "[" + s_to_string[severity] + "]" + "[" +
            datetime + "]: " + message;
    // Print to console and save the logs
    if (print_and_save) {
        print(log_string);
        write_to_file(log_string, save_file);
    }
}

//const std::map<Log::Severity, std::string> Log::s_to_string = {
//        {Log::emergency, "EMERGENCY"},
//        {Log::alert, "ALERT"},
//        {Log::critical, "CRITICAL"},
//        {Log::error, "ERROR"},
//        {Log::warning, "WARNING"},
//        {Log::notice, "NOTICE"},
//        {Log::info, "INFO"},
//        {Log::debug, "DEBUG"}
//};

Log::~Log() = default;
// std::cout << "Destroying Log." << std::endl;

void Log::write_to_file(const std::string& message, const std::string& filename) {
    std::ofstream file;
    file.open(filename);
    file << message + "\n";
    file.close();
}

void Log::print(const std::string& message) {
    std::cout << message << std::endl;
}

// Taken from https://stackoverflow.com/questions/27136854/c11-actual-system-time-with-milliseconds
// Getting time with millisecond precision
template<typename T>
std::string Log::get_time(std::chrono::time_point<T> time) {
    // using namespace std
    // using namespace std::chrono
    // Still better to be explicit rather than using namespaces
    std::time_t curr_time = T::to_time_t(time);
    char time_buffer[100];
    std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", localtime(&curr_time));
    typename T::duration since_epoch = time.time_since_epoch();
    std::chrono::seconds s = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
    since_epoch -= s;
    std::chrono::milliseconds milli = std::chrono::duration_cast<std::chrono::milliseconds>(since_epoch);
    std::stringstream milliseconds;
    milliseconds << std::setw(3) << std::setfill('0') << milli.count(); // Fill with leading zeros
    std::string formatted_time = std::string(time_buffer) + ":" + milliseconds.str();
    return formatted_time;
}
