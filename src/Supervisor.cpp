#include <vtol_tiltwing/Supervisor.hpp>
#include <vtol_tiltwing/ClockManager.hpp>
#include <vtol_tiltwing/IMUReadTask.hpp>
#include <vtol_tiltwing/RCReadTask.hpp>
#include <vtol_tiltwing/PIDControlTask.hpp>
#include <vtol_tiltwing/MotorActuateTask.hpp>
#include "Logger.hpp"

#include <string>

// Default log severity
LogSeverity Logger::printSeverity = LogSeverity::DEBUG;

Supervisor::Supervisor(){
    registry = new Registry();
    flag = new Flag();
    clockManager = ClockManager();
    imuReadTask = IMUReadTask();
    rcReadTask = RCReadTask();
    pidControlTask = PIDControlTask();
    motorActuateTask = MotorActuateTask();
}

void Supervisor::initialize() {
    Logger::log("Initializing supervisor", LogSeverity::DEBUG);
    clockManager.initialize();
}

void Supervisor::execute() {
    Logger::log("Running supervisor", LogSeverity::DEBUG);
    clockManager.execute();

    // Read Tasks
    imuReadTask.execute(registry);
    rcReadTask.execute(registry);

    // Control Tasks
    pidControlTask.execute(registry, flag);

    // Actuate Tasks
    motorActuateTask.execute(flag);
    // XBee ActuateTask();

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
