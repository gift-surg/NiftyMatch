#include "downsample.h"
#include "cudamath.h"
#include <helper_cuda.h>

template<typename DataType>
__global__ void downsample_2(DataType* __restrict__ result, const int result_width, const int result_height,
                             const DataType* __restrict__ source, const int source_width, const int source_height)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (x >= result_width || y >= result_height) return;

    const int d = y * result_width + x;
    const int s = (y * 2 * source_width) + (x * 2);
    result[d] = source[s];
}

template<typename DataType>
void downsample_by_2(DataType* result, const int result_width, const int result_height,
                     const DataType* source, const int source_width, const int source_height,
                     cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(result_width, blocks.x), DivUp(result_height, blocks.y));
    downsample_2<<<grid, blocks, 0, stream>>> (result, result_width, result_height,
                                               source, source_width, source_height);
    getLastCudaError("Downsampling kernel failed");
}

template void downsample_by_2<float>(float*, const int, const int, const float*, const int, const int, cudaStream_t);
template void downsample_by_2<uchar4>(uchar4*, const int, const int, const uchar4*, const int, const int, cudaStream_t);
