#include <vtol_tiltwing/Supervisor.hpp>
#include <vtol_tiltwing/ClockManager.hpp>
#include "Logger.hpp"

#include <string>

// Default log severity
LogSeverity Logger::printSeverity = LogSeverity::DEBUG;

Supervisor::Supervisor(){
    clockManager = ClockManager();
    registry = new Registry();
}

void Supervisor::initialize() {
    Logger::log("Initializing supervisor", LogSeverity::DEBUG);
    clockManager.initialize();
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

    // Update Loop variable in registry
    int num_loops;
    registry->get("core.num_loops", num_loops);
    registry->put("core.num_loops", num_loops + 1);
    Logger::log("Currently at loop count: " + to_string(num_loops), LogSeverity::DEBUG);    
}

void Supervisor::run(){
    while(true){
        this->execute();
    }
}
