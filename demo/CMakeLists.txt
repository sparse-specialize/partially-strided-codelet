
add_executable(spmv_demo
        FusionDemo.cpp
        SPMV_demo.cpp
        ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC} ${SPMV_CODELETS_SRC} ${SPTRSV_CODELETS_SRC}
        )

set(SPMV_DEMO_LIBS ${Sympiler_Libs})

if (MKL_FOUND)
    set(SPMV_DEMO_LIBS ${SPMV_DEMO_LIBS} ${MKL_LIBRARIES})
endif()

if (PAPI_FOUND)
    set(SPMV_DEMO_LIBS ${SPMV_DEMO_LIBS} ${PAPI_LIBRARIES})
endif()

target_link_libraries(spmv_demo
        ${SPMV_DEMO_LIBS})

add_executable(sptrsv_demo
        FusionDemo.cpp
        SPTRSV_demo.cpp
        ${SRC}
        ${TRCGEN}
        ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC}
        ${METIS_SOURCE_FILES} ${SPMV_CODELETS_SRC} ${SPTRSV_CODELETS_SRC}
        )
link_directories(${METIS_LIBRARY})

set(DEMO_LIBS lbc)

if (METIS_FOUND)
    set(DEMO_LIBS ${DEMO_LIBS} metis)
endif()

if (MKL_FOUND)
    set(DEMO_LIBS ${DEMO_LIBS} ${MKL_LIBRARIES})
endif()

if (PAPI_FOUND)
    set(DEMO_LIBS ${DEMO_LIBS} ${PAPI_LIBRARIES})
endif()

target_link_libraries(sptrsv_demo
        ${DEMO_LIBS}
        )
add_definitions(-DCSV_LOG)
