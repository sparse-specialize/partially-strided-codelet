
add_executable(spmv_demo
        FusionDemo.cpp
        SPMV_demo.cpp
        ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC}
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
        ${METIS_SOURCE_FILES}
        )

if(METIS_FOUND)
    target_link_libraries(sptrsv_demo
            lbc
            ${METIS_LIBRARIES}
            )
else()
    target_link_libraries(sptrsv_demo
            lbc)
endif()
add_definitions(-DCSV_LOG)
