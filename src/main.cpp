#include <vtol_tiltwing/Supervisor.hpp>

int main(){
    Supervisor supervisor;
    supervisor.initialize();
    supervisor.run();
    return 0;
}
