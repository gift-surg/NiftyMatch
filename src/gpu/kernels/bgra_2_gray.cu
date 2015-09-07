#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "cudamath.h"

#include <iostream>

template<typename OutputType>
__global__ void grayscale(const uchar4* bgra, OutputType * output,
                          const int width, const int height)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(x < width && y < height) {
        const int i = y * width + x;
        output[i] = 0.07 * bgra[i].x + 0.72 * bgra[i].y + 0.21 * bgra[i].z;
    }
}

template<typename OutputType>
void cuda_grayscale(const uchar4 *bgra, OutputType *output, const int width, const int height,
                    cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x),  DivUp(height, blocks.y));
    grayscale <<<grid, blocks, 0, stream >>> (bgra, output, width, height);
    getLastCudaError("CUDA grayscale launch failed");
}

template void cuda_grayscale<float>(const uchar4*, float*, const int, const int,
                                    cudaStream_t);



template<typename OutputType>
__global__ void extract_channel(const uchar4* bgra, OutputType * output,
                                     const int width, const int height, const int channel)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(x < width && y < height) {
        const int i = y * width + x;
        if (channel == 0) output[i] = (OutputType)bgra[i].x; // Blue
        else if (channel == 1) output[i] = (OutputType)bgra[i].y; // Green
        else if (channel == 2) output[i] = (OutputType)bgra[i].z; // Red
        else if (channel == 3) output[i] = (OutputType)bgra[i].w; // Alpha
    }
}


template<typename OutputType>
void cuda_extract_channel(const uchar4* bgra, OutputType * output, const int width, const int height,
                          const int channel, cudaStream_t stream=0)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x),  DivUp(height, blocks.y));
    extract_channel <<<grid, blocks, 0, stream >>> (bgra, output, width, height, channel);
    getLastCudaError("CUDA extract channel launch failed");
}

template void cuda_extract_channel<float>(const uchar4*, float*, const int, const int,
                                          const int, cudaStream_t);


template<typename InputType>
__global__ void put_channel(uchar4* bgra, const InputType * input,
                            const int width, const int height, const int channel)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(x < width && y < height) {
        const int i = y * width + x;
        if (channel == 0) bgra[i].x = (unsigned char)input[i]; // Blue
        else if (channel == 1) bgra[i].y = (unsigned char)input[i]; // Green
        else if (channel == 2) bgra[i].z = (unsigned char)input[i]; // Red
        //else if (channel == 3) bgra[i].w = (unsigned char)input[i]; // Alpha
        else if (channel == 3) bgra[i].w = 255; // Alpha
    }
}

template<typename InputType>
void cuda_put_channel(uchar4* bgra, const InputType * input, const int width, const int height,
                      const int channel, cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x),  DivUp(height, blocks.y));
    put_channel <<<grid, blocks, 0, stream >>> (bgra, input, width, height, channel);
    getLastCudaError("CUDA put channel launch failed");
}

template void cuda_put_channel<float>(uchar4*, const float*, const int, const int,
                                      const int, cudaStream_t);


__global__ void set_alpha_to_const(uchar4* bgra, const int width, const int height, const unsigned char val)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(x < width && y < height) {
        const int i = y * width + x;
        bgra[i].w = val;
    }
}

void cuda_set_alpha_to_const(uchar4* bgra, const int width, const int height,
                             const unsigned char val, cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(width, blocks.x),  DivUp(height, blocks.y));
    set_alpha_to_const<<<grid, blocks, 0, stream >>> (bgra, width, height, val);
    getLastCudaError("CUDA set alpha launch failed");
}
