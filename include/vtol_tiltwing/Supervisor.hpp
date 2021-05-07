#include <vtol_tiltwing/ClockManager.hpp>
#include <vtol_tiltwing/Registry.hpp>
#include <vtol_tiltwing/Flag.hpp>

#ifndef VTOL_TILTWING_SUPERVISOR_HPP
#define VTOL_TILTWING_SUPERVISOR_HPP

class Supervisor {
private:
    ClockManager clockManager;
    Registry *registry;
    Flag *flag;
    IMUReadTask imuReadTask;
    RCReadTask rcReadTask;
    PIDControlTask pidControlTask;
    MotorActuateTask motorActuateTask;

public:
    Supervisor();
    void initialize();
    void execute();
    void run();
};

#endif //VTOL_TILTWING_SUPERVISOR_HPP
