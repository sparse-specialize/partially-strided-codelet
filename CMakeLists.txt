cmake_minimum_required(VERSION 3.16)
project(DDT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(DownloadProject)

download_project(PROJ sympiler
        GIT_REPOSITORY https://github.com/sympiler/sympiler.git
        PREFIX  ${CMAKE_CURRENT_LIST_DIR}/external/
        UPDATE_DISCONNECTED 1
        )
set(SYMPILER_LBC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/sympiler/lbc/")
#add_subdirectory(external/sympiler-src/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -march=native -fopenmp")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen")
include_directories("${CMAKE_CURRENT_LIST_DIR}/lib/cxxopts/include")

aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src" SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen" TRCGEN)

add_executable(DDT main.cpp ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC})

add_subdirectory(demo)
