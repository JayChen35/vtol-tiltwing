#include <chrono>

#ifndef VTOL_TILTWING_CLOCKMANAGER_HPP
#define VTOL_TILTWING_CLOCKMANAGER_HPP

using namespace std::chrono;

class ClockManager {
private:
    steady_clock::time_point last_mcl_start_time;
    bool mcl_has_started;
    int mil_per_cycle;

public:
    ClockManager();
    void initialize(int mil_per_cycle);
    void execute();
};



#endif //VTOL_TILTWING_CLOCKMANAGER_HPP
