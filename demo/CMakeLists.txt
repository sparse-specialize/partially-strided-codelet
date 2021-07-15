
include_directories(smp
        )
add_executable(spmv_demo
        FusionDemo.cpp
        SPMV_demo.cpp
        ${SRC} ${TRCGEN} ${MATRIX_PARSING_SRC}
        )
