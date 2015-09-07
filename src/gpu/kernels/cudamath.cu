#include "cudamath.h"
#include "helper_cuda.h"
#include <math.h>

int DivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

int DivDown(int a, int b)
{
    return a / b;
}

int AlignUp(int a, int b)
{
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

int AlignDown(int a, int b)
{
    return a - a % b;
}

template<typename TYPE>
__global__ void cuda_subtract_images(const TYPE* __restrict__ A, const TYPE* __restrict__ B,
                                     TYPE * __restrict__ result, const int width, const int height)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (x >= width || y >= height) return;
    const int index = y * width + x;
    result[index] = A[index] - B[index];
}

template<typename TYPE>
__global__ void cuda_compute_gradient(const TYPE* __restrict__ source, float2* __restrict__ grad,
                                      const int width, const int height)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) return;

    TYPE nx = source[(y * width) + x + 1]; TYPE px = source[(y * width) + x - 1];
    TYPE ny = source[(y + 1) * width + x]; TYPE py = source[(y - 1) * width + x];

    float dx = nx - px;
    float dy = ny - py;
    float g = 0.5 * sqrtf(dx * dx  + dy * dy);
    float r = (g == 0.0f? 0.0f : mod_2pi_f(atan2(dy, dx) + 2 * M_PI));
    grad[y * width + x] = make_float2(g, r);
}

template<typename TYPE>
void subtract(const TYPE* __restrict__ A, const TYPE* __restrict__ B,
              TYPE * __restrict__ result, const int width, const int height,
              cudaStream_t stream)
{
    checkCudaErrors(cudaFuncSetCacheConfig(cuda_subtract_images<TYPE>,
                                           cudaFuncCachePreferL1));
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x), DivUp(height, blocks.y));
    cuda_subtract_images<<<grid, blocks, 0, stream>>> (A, B, result, width, height);
    getLastCudaError("Subtract launch failed");
}

template void subtract<float>(const float*, const float*, float*, const int, const int, cudaStream_t);

template<typename TYPE>
void gradient(const TYPE * source, float2 * result, const int width, const int height, cudaStream_t stream)
{
    checkCudaErrors(cudaFuncSetCacheConfig(cuda_subtract_images<TYPE>,
                                           cudaFuncCachePreferL1));
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x), DivUp(height, blocks.y));
    cuda_compute_gradient<<<grid, blocks, 0, stream>>> (source, result, width, height);
    getLastCudaError("Set gradient launch failed");
}
template void gradient<float>(const float* source, float2* result, const int width, const int height, cudaStream_t);
