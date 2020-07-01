#include <vtol_tiltwing/mcl/Supervisor.hpp>
#include <vtol_tiltwing/mcl/ClockManager.hpp>

Supervisor::Supervisor(Logger* logger){
    this->_logger = logger;
    this->clockManager = ClockManager();
}

void Supervisor::initialize() {
    i = 10;
    this->clockManager.initialize(1000);
}

void Supervisor::execute() {
    this->_logger->log("Running MCL", Log::Severity::debug);
    this->clockManager.execute();
}

void Supervisor::run(){
    while(true){
        this->execute();
    }
}

