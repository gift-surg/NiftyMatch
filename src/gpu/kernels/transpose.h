#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

template<typename TYPE>
void transpose(TYPE * odata,
               const TYPE * idata,
               int width, int height,
               cudaStream_t stream = 0);

#endif
