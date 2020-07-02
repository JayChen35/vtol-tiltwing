#include <vtol_tiltwing/mcl/Registry.hpp>

Registry::Registry() {
    // Add elements to the registry here
    add<int>("core.num_loops", 0);
}
