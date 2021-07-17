cmake_minimum_required(VERSION 3.16)
project(DDT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -march=native -fopenmp")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen")
include_directories("${CMAKE_CURRENT_LIST_DIR}/lib/cxxopts/include")
set(SYMPILER_LBC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/")
set(SYMPILER_UTIL "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/utils/")
set(SYMPILER_UTIL_INC "${SYMPILER_UTIL}/includes/")
include_directories(${SYMPILER_LBC_ROOT}/src/includes/
        ${SYMPILER_UTIL}/external/includes/
        ${SYMPILER_UTIL_INC})
set(Sympiler_Libs
        "lbc;metis;")
set(Sympiler_SRC
        "${SYMPILER_UTIL}/BCSCMatrix.cpp"
        "${SYMPILER_UTIL}/sparse_io.cpp"
        "${SYMPILER_UTIL}/metis_interface.cpp"
        "${SYMPILER_UTIL}/external/mmio.cpp")


aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src" SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen" TRCGEN)

add_executable(DDT main.cpp ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC})
target_link_libraries(DDT
        ${Sympiler_Libs})

add_subdirectory(lib/lbc/)
add_subdirectory(demo)
