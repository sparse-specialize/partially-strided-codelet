
add_executable(spmv_demo
        FusionDemo.cpp
        SPMV_demo.cpp
        ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC} ${SPMV_CODELETS_SRC} ${SPTRSV_CODELETS_SRC}
        )
target_link_libraries(spmv_demo
        ${Sympiler_Libs})


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

# TODO: Fix this hack
if(METIS_FOUND)
    target_link_libraries(sptrsv_demo
            lbc
            metis
            )
else()
    target_link_libraries(sptrsv_demo
            lbc)
endif()
add_definitions(-DCSV_LOG)
