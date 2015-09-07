SET(NAME kernels)
SET(HEADERS
    bgra_2_gray.h
    transpose.h
    match.h
    dog.h
    cudamath.h
    resample.h
    downsample.h
    convolution.h
    ransac.h
    keypoint.h
    orientation.h
    descriptor.h)
CUDA_ADD_LIBRARY(${NAME}
    ${HEADERS}
    bgra_2_gray.cu
    transpose.cu
    match.cu
    dog.cu
    cudamath.cu
    resample.cu
    downsample.cu
    convolution.cu
    ransac.cu
    keypoint.cu
    orientation.cu
    descriptor.cu)
INSTALL(FILES
    ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
INSTALL(TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)