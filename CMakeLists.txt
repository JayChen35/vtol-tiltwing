cmake_minimum_required(VERSION 3.16)
project(vtol_tiltwing)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_FILES src/main.cpp
        lib/Logger/src/Log.cpp
        lib/Logger/src/Logger.cpp src/mcl/Supervisor.cpp include/vtol_tiltwing/mcl/Supervisor.hpp include/vtol_tiltwing/mcl/ClockManager.hpp src/mcl/ClockManager.cpp)

include_directories(include)
include_directories(lib/Logger/include)

add_executable(vtol_tiltwing ${SOURCE_FILES})
# add_library(logger_library STATIC lib/Logger/src/Log.cpp lib/Logger/include/Log.hpp lib/Logger/src/Logger.cpp lib/Logger/include/Logger.hpp)

find_library(LOGGER_LIBRARY logger_library lib)
target_link_libraries(vtol_tiltwing LINK_PUBLIC ${LOGGER_LIBRARY})
