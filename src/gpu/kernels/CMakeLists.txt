SET(NAME kernels)
CUDA_ADD_LIBRARY(${NAME}
    bgra_2_gray.h bgra_2_gray.cu
    transpose.h transpose.cu
    match.h match.cu
    dog.h dog.cu
    cudamath.h cudamath.cu
    resample.h resample.cu
    downsample.h downsample.cu
    convolution.h convolution.cu
    ransac.h ransac.cu
    keypoint.h keypoint.cu
    orientation.h orientation.cu
    descriptor.h descriptor.cu)
INSTALL(TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
