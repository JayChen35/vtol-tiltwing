#include <vtol_tiltwing/Registry.hpp>

Registry::Registry() {
    // Add elements to the registry here
    put("core.num_loops", 0);
    // RC Inputs
    put("rc_input.throttle", (float)0.0); // Throttle (0, 100)
    put("rc_input.pitch", (float)0.0); // Pitch (-100, 100)
    put("rc_input.roll", (float)0.0); // Roll (-100, 100)
    put("rc_input.yaw", (float)0.0); // Yaw (-100, 100)
}
