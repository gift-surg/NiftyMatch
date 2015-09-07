#ifndef __CONVOLUTION_H__
#define __CONVOLUTION_H__

#include <cuda.h>
#include <cuda_runtime.h>

template<typename TYPE>
void convolve(TYPE* result, const TYPE* image, TYPE* buffer,
              const int width, const int height, const float* kernel,
              const int kernel_radius, cudaStream_t stream=0);

#endif
