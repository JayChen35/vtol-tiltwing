#include <vtol_tiltwing/ClockManager.hpp>

#ifndef VTOL_TILTWING_SUPERVISOR_HPP
#define VTOL_TILTWING_SUPERVISOR_HPP

class Supervisor {
private:
    ClockManager clockManager;

public:
    Supervisor();
    void initialize();
    void execute();
    void run();
};

#endif //VTOL_TILTWING_SUPERVISOR_HPP
