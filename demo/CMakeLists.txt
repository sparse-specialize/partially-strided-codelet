
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
        ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC}
        ${Sympiler_SRC}
        )
target_link_libraries(sptrsv_demo
        ${Sympiler_Libs}
        metis)
add_definitions(-DMETIS)
