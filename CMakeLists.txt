cmake_minimum_required(VERSION 3.16)
project(DDT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -march=native -fopenmp")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen")
include_directories("${CMAKE_CURRENT_LIST_DIR}/lib/cxxopts/include")

aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src" SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen" TRCGEN)

add_executable(DDT main.cpp ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC})

add_subdirectory(demo)
