SET(NAME sift)
CUDA_ADD_LIBRARY(${NAME}
    siftparams.h
    pyramidata.h pyramidata.cu
    siftdata.h siftdata.cu
    siftfunctions.h siftfunctions.cu)
TARGET_LINK_LIBRARIES(${NAME} kernels)
INSTALL(TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
