#include <vtol_tiltwing/Flag.hpp>

Flag::Flag() {
    // Add elements to the flag here
    put("motor.throttle", (float)0.0); // Throttle (0, 100)
    put("motor.pitch", (float)0.0); // Pitch (-100, 100)
    put("motor.roll", (float)0.0); // Roll (-100, 100)
    put("motor.yaw", (float)0.0); // Yaw (-100, 100)
}
