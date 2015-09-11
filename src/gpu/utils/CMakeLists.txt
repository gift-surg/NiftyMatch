SET(NAME gpuutils)
SET(HEADERS
    cudatimer.h
    cudatex2D.h
    cudautils.h
    exception.h)
CUDA_ADD_LIBRARY(${NAME}
    ${HEADERS}
    cudatimer.cu
    stridedrange.cu
    cudatex2D.cu
    cudautils.cpp)
TARGET_LINK_LIBRARIES(${NAME} ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
INSTALL_INCLUDES(${HEADERS})
INSTALL_BINARIES(${NAME})
