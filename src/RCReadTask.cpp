#include <vtol_tiltwing/RCReadTask.hpp>
#include <constants.hpp>
#include <helpers.hpp>
#include "Arduino.h"


RCReadTask::RCReadTask(){
    pinMode(THROTTLE_PIN, INPUT);
    pinMode(PITCH_PIN, INPUT);
    pinMode(ROLL_PIN, INPUT);
    pinMode(YAW_PIN, INPUT);
}

int RCReadTask::readChannel(int pin){
    #ifndef DESKTOP
        #include "Arduino.h"
        int val = pulseIn(pin, HIGH);
        return val;
    #endif
}

void RCReadTask::execute(Registry *registry){
    float throttleRead = (float)readChannel(THROTTLE_PIN);
    float pitchRead = (float)readChannel(PITCH_PIN);
    float rollRead = (float)readChannel(ROLL_PIN);
    float yawRead = (float)readChannel(YAW_PIN);
    float throttleInput = map_value(throttleRead, 1000.0, 2000.0, 0.0, 100.0);
    float pitchInput = map_value(pitchRead, 1000.0, 2000.0, 0.0, 100.0);
    float rollInput = map_value(rollRead, 1000.0, 2000.0, 0.0, 100.0);
    float yawInput = map_value(yawRead, 1000.0, 2000.0, 0.0, 100.0);

    registry->put("rc_input.throttle", throttleInput);
    registry->put("rc_input.pitch", pitchInput);
    registry->put("rc_input.roll", rollInput);
    registry->put("rc_input.yaw", yawInput);

}

