cmake_minimum_required(VERSION 3.1)
project(DDT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -march=native -fopenmp")

set(SYMPILER_LBC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/")
set(SYMPILER_UTIL "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/utils/")
set(SPARSE_UTIL_SRC "${CMAKE_CURRENT_SOURCE_DIR}/lib/lbc/utils/")
set(SYMPILER_UTIL_INC "${SYMPILER_UTIL}/includes/")
set(MIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/")
set(CSR5_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/CSR5_LOCAL/CSR5_avx2/" "${CMAKE_CURRENT_SOURCE_DIR}/lib/CSR5_LOCAL/CSR5_avx2/detail" "${CMAKE_CURRENT_SOURCE_DIR}/lib/CSR5_LOCAL/CSR5_avx2/detail/avx2")

find_package(METIS OPTIONAL_COMPONENTS)
find_package(MKL OPTIONAL_COMPONENTS)
find_package(PAPI OPTIONAL_COMPONENTS)

if(NOT METIS_FOUND)
    message(STATUS  "METIS not detected, downloading it..." )
    set(SHARED  TRUE CACHE BOOL "shared library" FORCE)
    set(METIS_INCLUDE_DIRS ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include)
    set(METIS_INCLUDES ${SYMPILER_LBC_ROOT}/external/metis/metis_shared-src/include)
    set(METIS_LIBRARY_DIRS ${SYMPILER_LBC_ROOT}/lib/)
    set(METIS_LIBRARIES metis)
    set(METIS_FOUND TRUE)
    message(STATUS "METIS library is found at ${METIS_LIBRARY}")
endif()

if(METIS_FOUND)
    add_definitions(-DMETIS)
    set(METIS_SOURCE_FILES "${SPARSE_UTIL_SRC}/metis_interface.cpp")
endif()

if (MKL_FOUND)
  add_definitions(-DMKL)
endif()

if (PAPI_FOUND)
    message('nice')
    add_definitions(-DPAPI)
endif()

if ($ENV{PROFILE})
    add_definitions(-DPROFILE)
    include_directories(
            ${PAPI_INCLUDE_DIRS}
    )
endif()


include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
include_directories("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen")
include_directories("${CMAKE_CURRENT_LIST_DIR}/lib/cxxopts/include")
include_directories(
        ${SYMPILER_LBC_ROOT}/src/includes/
        ${SYMPILER_UTIL}/external/includes/
        ${SYMPILER_UTIL_INC}
        ${METIS_INCLUDE_DIRS}
        ${METIS_INCLUDES}
        ${MKL_INCLUDE_DIR}
 #       ${PAPI_INCLUDE_DIRS}
        ${MIS_DIR}
        )

set(Sympiler_Libs
        lbc
       ${METIS_LIBRARIES} )

set(Sympiler_SRC
        "${SYMPILER_UTIL}/BCSCMatrix.cpp"
        "${SYMPILER_UTIL}/sparse_io.cpp"
        "${SYMPILER_UTIL}/metis_interface.cpp"
        "${SYMPILER_UTIL}/external/mmio.cpp"
        "${MIS_DIR}/lbc_csc_dag.cpp")


aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src" SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/spmv_codelets/" SPMV_CODELETS_SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/sptrsv_codelets/" SPTRSV_CODELETS_SRC)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/trace_gen" TRCGEN)
aux_source_directory("${CMAKE_CURRENT_LIST_DIR}/src/code_gen" CODE_GEN)

add_executable(DDT main.cpp ${SRC} ${TRCGEN} ${CODE_GEN} ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC} ${SPMV_CODELETS_SRC} ${SPTRSV_CODELETS_SRC})
target_link_libraries(DDT
        ${Sympiler_Libs})

add_subdirectory(lib/lbc/)
add_subdirectory(demo)
add_subdirectory(codelet_demo)
