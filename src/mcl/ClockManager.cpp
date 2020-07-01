//
// Created by Srikar on 7/1/2020.
//

#include <vtol_tiltwing/mcl/ClockManager.hpp>
#include <iostream>

ClockManager::ClockManager() {
    mcl_has_started = false;
    last_mcl_start_time = chrono::steady_clock::now();
}

void ClockManager::initialize(int mil_per_cycle){
    this->mil_per_cycle = mil_per_cycle;
}

void ClockManager::execute(){
    if(mcl_has_started){
        // Wait for that time to pass
        while(chrono::steady_clock::now() - last_mcl_start_time < chrono::milliseconds {this->mil_per_cycle}){}
    }
    mcl_has_started = true;
    last_mcl_start_time = chrono::steady_clock::now();
}

