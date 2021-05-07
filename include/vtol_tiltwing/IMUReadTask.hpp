#ifndef VTOL_TILTWING_IMUREADTASK_HPP
#define VTOL_TILTWING_IMUREADTASK_HPP

#include <vtol_tiltwing/Registry.hpp>
#include <Adafuit_BNO055/Adafruit_BNO055.h>
#include <constants.hpp>

class IMUReadTask {
private:
    Adafruit_BNO055 bno;

public:
    IMUReadTask()
        : bno(BNO_ID, BNO_ADDRESS)
        { if(!bno.begin()){
            Logger::log("BNO NOT DETECTED!!", LogSeverity::ERROR);
            while(1);
        } }
    void execute(Registry *registry);
};


#endif // VTOL_TILTWING_IMUREADTASK_HPP
