#include <vtol_tiltwing/MotorActuateTask.hpp>
#include <Logger.hpp>
#include <constants.hpp>
#include <helpers.hpp>
#include "Arduino.h"

MotorActuateTask::MotorActuateTask(){
    throttle.attach(THROTTLE_OUT);
    pitch.attach(PITCH_OUT);
    roll.attach(ROLL_OUT);
    yaw.attach(YAW_OUT);    
}

void MotorActuateTask::execute(Flag *flag){
    float throttle_out, pitch_out, roll_out, yaw_out;
    flag->get("motor.throttle", throttle_out);
    flag->get("motor.pitch", pitch_out);
    flag->get("motor.roll", roll_out);
    flag->get("motor.yaw", yaw_out);
    Logger::log("Writing values to motors: " + to_string(throttle_out) + ", " + to_string(pitch_out) + ", " + to_string(roll_out) = ", " + to_string(yaw_out));

    throttle_out = map_value(throttle_out, 0.0, 100.0, 1000.0, 2000.0);
    pitch_out = map_value(pitch_out, -100.0, 100.0, 0.0, 180.0);
    roll_out = map_value(roll_out, -100.0, 100.0, 0.0, 180.0);
    yaw_out = map_value(yaw_out, -100.0, 100.0, 0.0, 180.0);

    throttle.writeMicroseconds(throttle_out);
    pitch.write(pitch_out);
    roll.write(roll_out);
    yaw.write(yaw_out);
}
