#include <Logger.hpp>
#include <vtol_tiltwing/mcl/ClockManager.hpp>

#ifndef VTOL_TILTWING_SUPERVISOR_HPP
#define VTOL_TILTWING_SUPERVISOR_HPP

class Supervisor {
private:
    int i;
    Logger* _logger;
    ClockManager clockManager;

public:
    Supervisor(Logger* logger);
    void initialize();
    void execute();
    void run();
};


#endif //VTOL_TILTWING_SUPERVISOR_HPP
