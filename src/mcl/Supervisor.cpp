#include <vtol_tiltwing/mcl/Supervisor.hpp>
#include <vtol_tiltwing/mcl/ClockManager.hpp>

Supervisor::Supervisor(Logger* logger){
    this->logger = logger;
    clockManager = ClockManager();
    Registry temp = Registry();
    registry = &temp;
}

void Supervisor::initialize() {
    i = 10;
    clockManager.initialize(50, 10);
}

void Supervisor::execute() {
    logger->log("Running supervisor", Log::Severity::debug);
    int num_loops = registry->get<int>("core.num_loops");
    logger->log("Currently at loop count: " + num_loops, Log::Severity::debug);
    clockManager.execute();
}

void Supervisor::run(){
    while(true){
        this->execute();
    }
}
