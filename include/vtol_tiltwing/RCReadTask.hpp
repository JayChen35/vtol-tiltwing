#ifndef VTOL_TILTWING_RCREADTASK_HPP
#define VTOL_TILTWING_RCREADTASK_HPP

#include <vtol_tiltwing/Registry.hpp>

class RCReadTask {
private:
    int readChannel(int channel);

public:
    RCReadTask();
    void execute(Registry *registry);
};


#endif // VTOL_TILTWING_RCREADTASK_HPP
