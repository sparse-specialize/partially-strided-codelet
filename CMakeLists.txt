cmake_minimum_required(VERSION 3.16)
project(DDT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -march=native -fopenmp")

set(SYMPILER_LBC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/")
set(SYMPILER_UTIL "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/utils/")
set(SPARSE_UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/utils/")
set(SYMPILER_UTIL_INC "${SYMPILER_UTIL}/includes/")

find_package(METIS OPTIONAL_COMPONENTS)
if(NOT METIS_FOUND)
    message(STATUS  "METIS not detected, downloading it..." )
    #    find_program(MAKE_EXE NAMES gmake nmake make)
    #    set(metis_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/metis")
    #    set(metis_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${metis_INSTALL_DIR})
    #set(EXTERNAL_INSTALL_LOCATION ${CMAKE_CURRENT_LIST_DIR}/external)
    # set(GKLIB_PATH ${SYMPILER_EXTERNAL}/metis/metis-src/GKLIB/ CACHE PATH "d")

    set(SHARED  TRUE CACHE BOOL "shared library" FORCE)
    set(METIS_INCLUDE_DIRS ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include)
    set(METIS_INCLUDES ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include)
    set(METIS_LIBRARY_DIRS ${SYMPILER_LBC_ROOT}/lib/)
    set(METIS_LIBRARIES ${SYMPILER_LBC_ROOT}/lib/libmetis.so)
    set(METIS_FOUND TRUE)
    message(STATUS "METIS library is found at ${METIS_LIBRARY}")
endif()
if(METIS_FOUND)
    add_definitions(-DMETIS)
    set(METIS_SOURCE_FILES "${SPARSE_UTIL_SRC}/metis_interface.cpp")
    #message("===> ${METIS_SOURCE_FILES}")
endif()


include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen")
include_directories("${CMAKE_CURRENT_LIST_DIR}/lib/cxxopts/include")
include_directories(${SYMPILER_LBC_ROOT}/src/includes/
        ${SYMPILER_UTIL}/external/includes/
        ${SYMPILER_UTIL_INC}
        ${METIS_INCLUDE_DIRS} ${METIS_INCLUDES}
        )

set(Sympiler_Libs
        "lbc;metis;")

set(Sympiler_SRC
        "${SYMPILER_UTIL}/BCSCMatrix.cpp"
        "${SYMPILER_UTIL}/sparse_io.cpp"
        "${SYMPILER_UTIL}/metis_interface.cpp"
        "${SYMPILER_UTIL}/external/mmio.cpp")


aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src" SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/spmv_codelets/" SPMV_CODELETS_SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/sptrsv_codelets/" SPTRSV_CODELETS_SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen" TRCGEN)

add_executable(DDT main.cpp ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC} ${SPMV_CODELETS_SRC} ${SPTRSV_CODELETS_SRC})
target_link_libraries(DDT
        ${Sympiler_Libs})

add_subdirectory(lib/lbc/)
add_subdirectory(demo)
