#include <vtol_tiltwing/mcl/ClockManager.hpp>
#include "iostream"

ClockManager::ClockManager() {
    mcl_has_started = false;
    last_mcl_start_time = steady_clock::now();
}

void ClockManager::initialize(int mil_per_cycle, int min_mil_per_cycle){
    this->mil_per_cycle = mil_per_cycle;
    this->min_mil_per_cycle = min_mil_per_cycle;
    mcl_initialize_time = steady_clock::now();
    num_loops = 0;
}

void ClockManager::execute(){
    num_loops += 1;
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

