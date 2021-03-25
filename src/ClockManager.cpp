#include <vtol_tiltwing/ClockManager.hpp>
#include <Logger.hpp>
#include <constants.hpp>
#include "iostream"

ClockManager::ClockManager() {
    mcl_has_started = false;
    last_mcl_start_time = steady_clock::now();
    this->mil_per_cycle = MIL_PER_CYCLE;
    this->min_mil_per_cycle = MIN_MIL_PER_CYCLE;
}

void ClockManager::initialize(){
    mcl_initialize_time = steady_clock::now();
    num_loops = 0;
    Logger::log("Initialized clock manager", LogSeverity::DEBUG);
}

void ClockManager::execute(){
    num_loops += 1;
    Logger::log("Executing clock manager", LogSeverity::DEBUG);
    if(mcl_has_started){
        // Wait for that time to pass, also takes into consideration average time
        while(steady_clock::now() - last_mcl_start_time < milliseconds{min_mil_per_cycle} ||
        steady_clock::now() - mcl_initialize_time < milliseconds{mil_per_cycle} * num_loops){}
    }
    else{
        mcl_initialize_time = steady_clock::now();
    }
    mcl_has_started = true;
    last_mcl_start_time = steady_clock::now();
}

