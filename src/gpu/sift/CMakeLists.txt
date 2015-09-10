SET(NAME sift)
SET(HEADERS
    siftparams.h
    pyramidata.h
    siftdata.h
    siftfunctions.h)
CUDA_ADD_LIBRARY(${NAME}
    ${HEADERS}
    pyramidata.cu
    siftdata.cu
    siftfunctions.cu)
TARGET_LINK_LIBRARIES(${NAME} kernels)
INSTALL_INCLUDES(${HEADERS})
INSTALL_BINARIES(${NAME})
