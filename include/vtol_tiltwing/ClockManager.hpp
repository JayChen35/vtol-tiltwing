#include <chrono>

#ifndef VTOL_TILTWING_CLOCKMANAGER_HPP
#define VTOL_TILTWING_CLOCKMANAGER_HPP

using namespace std::chrono;

class ClockManager {
private:
    steady_clock::time_point last_mcl_start_time;
    steady_clock::time_point mcl_initialize_time;
    bool mcl_has_started;
    long mil_per_cycle;
    long min_mil_per_cycle;
    long num_loops;

public:
    ClockManager();
    void initialize();
    void execute();
};

#endif //VTOL_TILTWING_CLOCKMANAGER_HPP
