#ifndef HELPER_LOGGER_HPP
#define HELPER_LOGGER_HPP

#include <string>
#include <iostream>

using namespace std;

enum LogSeverity {
    ERROR = 1,
    WARNING = 2,
    INFO = 3,
    DEBUG = 4
};

class Logger {
private:

    static string severityToString(LogSeverity severity){
        if(severity == ERROR){
            return "ERROR:";
        }
        if(severity == WARNING){
            return "WARNING:";
        }
        if(severity == INFO){
            return "INFO:";
        }
        return "DEBUG:";
    }

public:

    static LogSeverity printSeverity;

    Logger(){}

    static void setLogSeverity(LogSeverity severity){
        printSeverity = severity;
    }

    static void log(string msg, LogSeverity severity){
        if(severity <= printSeverity){
            // TODO: Check if desktop or not to determine print method
            string toPrint = severityToString(severity) + " " + msg;
            #ifdef DESKTOP
                #include <iostream>
                cout << toPrint << endl;
            #else
                #include "Arduino.h"
                Serial.println(toPrint);
            #endif
        }
    }

};



#endif //HELPER_LOGGER_HPP
