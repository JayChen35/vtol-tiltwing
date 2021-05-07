#ifndef VTOL_TILTWING_MOTORACTUATETASK_HPP
#define VTOL_TILTWING_MOTORACTUATETASK_HPP

#include <vtol_tiltwing/Flag.hpp>
#include <Servo.h>

class MotorActuateTask {
private:
    Flag *flag;
    Servo throttle, pitch, roll, yaw;
public:
    MotorActuateTask();
    void execute(Flag *flag);
};

#endif // VTOL_TILTWING_MOTORACTUATETASK_HPP
