#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "cudamath.h"

#define BLOCK_DIM 16	// Threadblock size for matrix transposition

template <typename TYPE>
__global__ void transpose_kernel(TYPE* __restrict__ odata, const TYPE* __restrict__ idata,
                                 int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
    unsigned int xIndex = __mul24(blockIdx.x, BLOCK_DIM) + threadIdx.x;
    unsigned int yIndex = __mul24(blockIdx.y, BLOCK_DIM) + threadIdx.y;
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
    }
}

template <typename TYPE>
void transpose (TYPE *odata, const TYPE *idata, int width, int height, cudaStream_t stream)
{
    checkCudaErrors(cudaFuncSetCacheConfig(transpose_kernel<TYPE>, cudaFuncCachePreferShared));
    dim3 blocks (DivUp(width, BLOCK_DIM), DivUp(height,BLOCK_DIM));
    dim3 threads (BLOCK_DIM, BLOCK_DIM);
    transpose_kernel<TYPE><<<blocks, threads, 0, stream>>> (odata , idata , width , height);
    getLastCudaError("Transpose kernel failed");
}

template void transpose<float>(float *, const float *, int, int, cudaStream_t);
