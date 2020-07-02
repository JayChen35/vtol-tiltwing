#include "Logger.hpp"
#include <vtol_tiltwing/mcl/Supervisor.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main() {
    std::cout << "Running main program." << std::endl;
    Logger logger = Logger();
    logger.log("Logger successfully initialized.", Log::Severity::info);
    // JSON library test
    json test;
    test["engine"] = "Aphlex1B";
    test["max_speed"] = 1045;
    logger.log("Example JSON: " + test.dump(), Log::Severity::info);
    // Create Supervisor instance and run
    Supervisor supervisor = Supervisor(&logger);
    supervisor.initialize();
    logger.log("Supervisor successfully initialized.", Log::Severity::debug);
    supervisor.run();
}
