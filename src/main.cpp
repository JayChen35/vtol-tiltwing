#include <iostream>
#include <iterator>
#include <map>
#include "Logger.hpp"


int main() {
    std::cout << "Running main program." << std::endl;
    Logger logger = Logger();
    logger.log("Logger successfully initialized.", Log::Severity::info);
    // Test all Logger modes
    // std::map<Log::Severity, std::string>::iterator itr;
    // for (itr = Log::s_to_string.begin(); itr != Log::s_to_string.end(); ++itr) {
    //     logger.log("Testing " + itr->second + "severity.", itr->first);
    // }
    logger.log("Logger test successfully finished.", Log::Severity::debug);
}