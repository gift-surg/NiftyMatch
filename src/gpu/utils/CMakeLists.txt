SET(NAME gpuutils)
CUDA_ADD_LIBRARY(${NAME}
    cudatimer.h cudatimer.cu
    stridedrange.cu
    cudatex2D.h cudatex2D.cu
    cudautils.h cudautils.cpp
    exception.h)
TARGET_LINK_LIBRARIES(${NAME} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
INSTALL(TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
