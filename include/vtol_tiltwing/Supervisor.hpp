#include <vtol_tiltwing/ClockManager.hpp>
#include <vtol_tiltwing/Registry.hpp>

#ifndef VTOL_TILTWING_SUPERVISOR_HPP
#define VTOL_TILTWING_SUPERVISOR_HPP

class Supervisor {
private:
    ClockManager clockManager;
    Registry *registry;

public:
    Supervisor();
    void initialize();
    void execute();
    void run();
};

#endif //VTOL_TILTWING_SUPERVISOR_HPP
