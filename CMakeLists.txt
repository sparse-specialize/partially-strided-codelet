cmake_minimum_required(VERSION 3.16)
project(DDT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fopenmp")

include_directories("./include")
include_directories("./lib/cxxopts/include")

aux_source_directory("./src" SRC)


add_executable(DDT main.cpp ${SRC} ${MATRIX_PARSING_SRC})
