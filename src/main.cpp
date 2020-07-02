#include "Logger.hpp"
#include <vtol_tiltwing/mcl/Supervisor.hpp>


int main() {
    std::cout << "Running main program." << std::endl;
    Logger logger = Logger();
    logger.log("Logger successfully initialized.", Log::Severity::info);
    Supervisor supervisor = Supervisor(&logger);
    logger.log("Supervisor successfully initialized.", Log::Severity::debug);
    supervisor.run();
}
