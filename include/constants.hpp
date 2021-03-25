#define MIL_PER_CYCLE 50
#define MIN_MIL_PER_CYCLE 10

// Pins
#ifndef DESKTOP
    #include "Arduino.h"
    #define THROTTLE_PIN A0
    #define PITCH_PIN A1
    #define ROLL_PIN A2
    #define YAW_PIN A3
#endif
