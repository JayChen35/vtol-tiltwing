cmake_minimum_required(VERSION 3.16)
project(vtol_tiltwing)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_FILES src/main.cpp)
set(LIB_FILES lib/Logger/src/Log.cpp)

include_directories(include)
include_directories(lib/Logger/include)

add_executable(vtol_tiltwing ${SOURCE_FILES})
add_library(logger_library STATIC ${LIB_FILES})
