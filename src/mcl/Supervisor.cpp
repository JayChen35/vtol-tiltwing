#include <vtol_tiltwing/mcl/Supervisor.hpp>
#include <vtol_tiltwing/mcl/ClockManager.hpp>

Supervisor::Supervisor(Logger* logger){
    _logger = logger;
    clockManager = ClockManager();
}

void Supervisor::initialize() {
    i = 10;
    clockManager.initialize(50, 10);
}

void Supervisor::execute() {
    _logger->log("Running MCL on infinite loop.", Log::Severity::debug);
    clockManager.execute();
}

void Supervisor::run(){
    while(true){
        this->execute();
    }
}

