#include <vtol_tiltwing/Supervisor.hpp>
#include <vtol_tiltwing/ClockManager.hpp>
#include "Logger.hpp"

#include <string>

// Default log severity
LogSeverity Logger::printSeverity = LogSeverity::DEBUG;

Supervisor::Supervisor(){
    clockManager = ClockManager();
}

void Supervisor::initialize() {
    Logger::log("Initializing supervisor", LogSeverity::DEBUG);
    clockManager.initialize(50, 10);
}

void Supervisor::execute() {
    Logger::log("Running supervisor", LogSeverity::DEBUG);
    clockManager.execute();
    // Read Tasks
    // IMUReadTask();
    // RCReadTask();

    // Control Tasks
    // PIDControlTask();

    // Actuate Tasks
    // MotorActuateTask();
    // ServoActuateTask();

    // int num_loops = registry->get<int>("core.num_loops");
    // Logger::log("Currently at loop count: " + num_loops, LogSeverity::DEBUG);
}

void Supervisor::run(){
    while(true){
        this->execute();
    }
}
