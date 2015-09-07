#include "convolution.h"
#include "cudamath.h"

#include <helper_cuda.h>
#include "exception.h"

#include <stdio.h>

#define ROW_TILE_W 128
#define kernel_radius_ALIGNED 16

#define COLUMN_TILE_W 16
#define COLUMN_TILE_H 48

template<typename DataType>
__global__ void convolve_rows(DataType* __restrict__ result, const DataType* __restrict__ image,
                              const int width, const int height,
                              const float * __restrict__ kernel, const int kernel_radius)
{
    //Data cache
    extern __shared__ DataType data[];

    //Current tile and apron limits, relative to row start
    const int tile_start = __mul24(blockIdx.x, ROW_TILE_W);
    const int tile_end = tile_start + ROW_TILE_W - 1;
    const int apron_start = tile_start - kernel_radius;
    const int apron_end = tile_end + kernel_radius;

    //Clamp tile and apron limits by image borders
    const int tile_end_clamped = min(tile_end, width - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, width - 1);

    //Row start index in image[]
    const int row_start = __mul24(blockIdx.y, width);

    //Aligned apron start. Assuming width and ROW_TILE_W are multiples
    //of half-warp size, row_start + apron_start_aligned is also a
    //multiple of half-warp size, thus having proper alignment
    //for coalesced image[] read.
    const int apron_start_aligned = tile_start - kernel_radius_ALIGNED;

    const int load_pos = apron_start_aligned + threadIdx.x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(load_pos >= apron_start)
    {
        const int s_mem_pos = load_pos - apron_start;
        data[s_mem_pos] =
                ((load_pos >= apron_start_clamped) && (load_pos <= apron_end_clamped)) ?
                    image[row_start + load_pos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();

    const int write_pos = tile_start + threadIdx.x;

    //Assuming width and ROW_TILE_W are multiples of half-warp size,
    //row_start + tile_start is also a multiple of half-warp size,
    //thus having proper alignment for coalesced result[] write.

    if(write_pos <= tile_end_clamped){
        const int s_mem_pos = write_pos - apron_start;
        DataType sum = 0;
        for(int k = -kernel_radius; k <= kernel_radius; k++)
            sum += data[s_mem_pos + k] * kernel[kernel_radius - k];

        result[row_start + write_pos] = sum;
    }
}


template<typename DataType>
__global__ void convolve_cols(DataType* __restrict__ result, const DataType* __restrict__ image,
                              const int width, const int height, const float * __restrict__ kernel,
                              const int kernel_radius,
                              const int smemStride, const int gmemStride)
{
    //Data cache
    extern __shared__ DataType data[];

    //Current tile and apron limits, in rows
    const int tile_start = __mul24(blockIdx.y, COLUMN_TILE_H);
    const int tile_end = tile_start + COLUMN_TILE_H - 1;
    const int apron_start = tile_start - kernel_radius;
    const int apron_end = tile_end   + kernel_radius;

    //Clamp tile and apron limits by image borders
    const int tile_end_clamped = min(tile_end, height - 1);
    const int apron_start_clamped = max(apron_start, 0);
    const int apron_end_clamped = min(apron_end, height - 1);

    //Current column index
    const int columnStart = __mul24(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

    //Shared and global memory indices for current column
    int s_mem_pos = __mul24(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = __mul24(apron_start + threadIdx.y, width) + columnStart;


    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apron_start + threadIdx.y; y <= apron_end; y += blockDim.y){
        data[s_mem_pos] =
                ((y >= apron_start_clamped) && (y <= apron_end_clamped)) ?
                    image[gmemPos] : 0;
        s_mem_pos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();

    //Shared and global memory indices for current column
    s_mem_pos = __mul24(threadIdx.y + kernel_radius, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = __mul24(tile_start + threadIdx.y , width) + columnStart;

    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tile_start + threadIdx.y; y <= tile_end_clamped; y += blockDim.y)
    {
        float sum = 0;
        for(int k = -kernel_radius; k <= kernel_radius; k++)
            sum += data[s_mem_pos + __mul24(k, COLUMN_TILE_W)] * kernel[kernel_radius - k];

        result[gmemPos] = sum;
        s_mem_pos += smemStride;
        gmemPos += gmemStride;
    }
}


template<typename TYPE>
void convolve(TYPE* __restrict__ result, const TYPE *image, TYPE* __restrict__ buffer,
              const int width, const int height, const float* __restrict__ kernel,
              const int kernel_radius, cudaStream_t stream)
{
    dim3 blockGridRows(DivUp(width, ROW_TILE_W), height);
    dim3 blockGridColumns(DivUp(width, COLUMN_TILE_W), DivUp(height, COLUMN_TILE_H));
    dim3 threadBlockRows(kernel_radius_ALIGNED + ROW_TILE_W + kernel_radius);
    dim3 threadBlockColumns(COLUMN_TILE_W, 16);
    const int row_shared_mem_bytes = (kernel_radius + ROW_TILE_W + kernel_radius) * sizeof(float);
    convolve_rows<<<blockGridRows, threadBlockRows, row_shared_mem_bytes>>>(buffer, image, width,
                                                                            height, kernel, kernel_radius);
    getLastCudaError("Convolution rows failed");
    const int col_shared_mem_bytes = COLUMN_TILE_W * (kernel_radius + COLUMN_TILE_H + kernel_radius) * sizeof(float);

    convolve_cols<<<blockGridColumns, threadBlockColumns, col_shared_mem_bytes, stream>>>
                (result, buffer, width, height, kernel, kernel_radius, COLUMN_TILE_W * threadBlockColumns.y,
                 width * threadBlockColumns.y);
    getLastCudaError("Convolution column failed");
}

template void convolve<float>(float *, const float*, float*, const int, const int, const float*, const int,
                              cudaStream_t);
