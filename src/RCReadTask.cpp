#include <vtol_tiltwing/RCReadTask.hpp>
#include <constants.hpp>
#include <helpers.hpp>


RCReadTask::RCReadTask(){
    throttleInput = 0;
    rollInput = 0;
    pitchInput = 0;
    yawInput = 0;
    #ifndef DESKTOP
        #include "Arduino.h"
        pinMode(THROTTLE_PIN, INPUT);
        pinMode(PITCH_PIN, INPUT);
        pinMode(ROLL_PIN, INPUT);
        pinMode(YAW_PIN, INPUT);
    #endif
}

int RCReadTask::readChannel(int pin){
    #ifndef DESKTOP
        #include "Arduino.h"
        int val = pulseIn(pin, HIGH);
        return val;
    #endif
}

void RCReadTask::execute(Registry *registry){
    #ifdef DESKTOP
        // TODO: Figure out how to do this
    #else
        #include "Arduino.h"
        float val = (float)readChannel(THROTTLE_PIN);
        throttleInput = map(val, 1000.0, 2000.0, 0.0, 100.0);
    #endif

    registry->put("rc_input.throttle", throttleInput);
    registry->put("rc_input.pitch", pitchInput);
    registry->put("rc_input.roll", rollInput);
    registry->put("rc_input.yaw", yawInput);

}

