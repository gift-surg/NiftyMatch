#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "helper_cuda.h"
#include "cudamath.h"

__global__ void transform_and_blend(uchar4 * canvas, const int cw, const int ch,
                                    cudaTextureObject_t frame, const int fw, const int fh,
                                    const int nw, const int nh, const float* mat3x3,
                                    const int tx, const int ty, cudaTextureObject_t mask,
                                    float* canvas_wts, cudaTextureObject_t frame_wts)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    __shared__ float shared_t[9];

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int i = 0; i < 9; ++i) shared_t[i] = mat3x3[i];
    }

    __syncthreads();

    const int pos_x = x + tx;
    const int pos_y = y + ty;

    if (x >= nw || y >= nh || pos_x < 0 || pos_x >= cw || pos_y < 0 || pos_y >= ch) return;

    float x_p = shared_t[0] * x + shared_t[1] * y + shared_t[2];
    float y_p = shared_t[3] * x + shared_t[4] * y + shared_t[5];
    float s_p = shared_t[6] * x + shared_t[7] * y + shared_t[8];
    x_p /= s_p;
    y_p /= s_p;

    if (x_p >= fw || y_p >= fh) return;

    float4 res = tex2D<float4>(frame, x_p + 0.5f, y_p + 0.5f);
    float in_mask = tex2D<float>(mask, x_p + 0.5f, y_p + 0.5f);

    if (in_mask <= 0.5) return;

    float new_weight = tex2D<float>(frame_wts, x_p + 0.5f, y_p + 0.5f);
    const int index = pos_y * cw + pos_x;

    if (canvas_wts[index] == 0)
    {
        canvas[index].x = (unsigned char)(res.x * 255.9999f);
        canvas[index].y = (unsigned char)(res.y * 255.9999f);
        canvas[index].z = (unsigned char)(res.z * 255.9999f);
        canvas[index].w = 255;
        canvas_wts[index] = new_weight;
    }

    else
    {
        uchar4 current_value = canvas[index];
        float current_weight = canvas_wts[index];
        float sum_wts = current_weight + new_weight;

        canvas[index].x = (unsigned char)((res.x * new_weight * 255.9999f + current_value.x * current_weight) / sum_wts);
        canvas[index].y = (unsigned char)((res.y * new_weight * 255.9999f + current_value.y * current_weight) / sum_wts);
        canvas[index].z = (unsigned char)((res.z * new_weight * 255.9999f + current_value.z * current_weight) / sum_wts);
        canvas[index].w = 255;
        canvas_wts[index] += new_weight;
    }
}

__global__ void resample_mask_2D(unsigned char* __restrict__ result, cudaTextureObject_t text,
                                 const int width, const int height,
                                 const float* __restrict__ x, const float* __restrict__ y,
                                 const float lower_limit)
{
    const int _x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int _y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (_x < width && _y < height) {
        const int i = _y * width + _x;
        float res = tex2D<float>(text, x[i] + 0.5f, y[i] + 0.5f);
        if (res <= lower_limit) result[i] = 0;
        else result[i] = res * 255.999f;
    }
}

__global__ void resample_2D(uchar4* __restrict__ result, cudaTextureObject_t text,
                            const int width, const int height,
                            const float* __restrict__ x, const float* __restrict__ y)
{
    const int _x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int _y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    if (_x < width && _y < height) {
        const int i = _y * width + _x;
        float4 res = tex2D<float4>(text, x[i] + 0.5f, y[i] + 0.5f);
        result[i].x = (unsigned char)(res.x * 255.9999f);
        result[i].y = (unsigned char)(res.y * 255.9999f);
        result[i].z = (unsigned char)(res.z * 255.9999f);
        result[i].w = (unsigned char)(res.w * 255.9999f);
    }
}

// Apply the inverse of the supplied perspective transform to generate new coordinates
template<typename TYPE>
__global__ void apply_perspective_inverse(TYPE * __restrict__ x_pos, TYPE * __restrict__ y_pos,
                                          const float * mat3x3, const int width, const int height)
{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    __shared__ float shared_t[9];
    __shared__ float shared_t_inv[9];

    // Let the first thread of each block bring the transform matrix
    // to shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int i = 0; i < 9; ++i) shared_t[i] = mat3x3[i];
        // Compute the inverse
        TYPE det = shared_t[0] * (shared_t[4] * shared_t[8] - shared_t[7] * shared_t[5]) -
                   shared_t[1] * (shared_t[3] * shared_t[8] - shared_t[5] * shared_t[6]) +
                   shared_t[2] * (shared_t[3] * shared_t[7] - shared_t[4] * shared_t[6]);

        TYPE invdet = 1/det;
        shared_t_inv[0] = (shared_t[4] * shared_t[8] - shared_t[7] * shared_t[5]) * invdet;
        shared_t_inv[1] = (shared_t[2] * shared_t[7] - shared_t[1] * shared_t[8]) * invdet;
        shared_t_inv[2] = (shared_t[1] * shared_t[5] - shared_t[2] * shared_t[4]) * invdet;
        shared_t_inv[3] = (shared_t[5] * shared_t[6] - shared_t[3] * shared_t[8]) * invdet;
        shared_t_inv[4] = (shared_t[0] * shared_t[8] - shared_t[2] * shared_t[6]) * invdet;
        shared_t_inv[5] = (shared_t[3] * shared_t[2] - shared_t[0] * shared_t[5]) * invdet;
        shared_t_inv[6] = (shared_t[3] * shared_t[7] - shared_t[6] * shared_t[4]) * invdet;
        shared_t_inv[7] = (shared_t[6] * shared_t[1] - shared_t[0] * shared_t[7]) * invdet;
        shared_t_inv[8] = (shared_t[0] * shared_t[4] - shared_t[3] * shared_t[1]) * invdet;
    }

    __syncthreads();

    if (x < width && y < height)
    {
        const int index = y * width + x;
        TYPE x_p = shared_t_inv[0] * x + shared_t_inv[1] * y + shared_t_inv[2];
        TYPE y_p = shared_t_inv[3] * x + shared_t_inv[4] * y + shared_t_inv[5];
        TYPE s_p = shared_t_inv[6] * x + shared_t_inv[7] * y + shared_t_inv[8];
        // Map back to the 2D plane by perspective divide.
        x_pos[index] = x_p/s_p;
        y_pos[index] = y_p/s_p;
    }
}

// Apply the supplied perspective transform to generate new coordinates
template<typename TYPE>
__global__ void apply_perspective(TYPE * __restrict__ x_pos, TYPE * __restrict__ y_pos,
                                  const float * mat3x3, const int width, const int height)

{
    const int x = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int y = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;

    __shared__ float shared_t[9];

    // Let the first thread of each block bring the transform matrix
    // to shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        for (int i = 0; i < 9; ++i) shared_t[i] = mat3x3[i];
    }

    __syncthreads();

    if (x < width && y < height)
    {
        const int index = y * width + x;
        TYPE x_p = shared_t[0] * x + shared_t[1] * y + shared_t[2];
        TYPE y_p = shared_t[3] * x + shared_t[4] * y + shared_t[5];
        TYPE s_p = shared_t[6] * x + shared_t[7] * y + shared_t[8];
        // Map back to the 2D plane by perspective divide.
        x_pos[index] = x_p/s_p;
        y_pos[index] = y_p/s_p;
    }
}

void resample_perspective_transform(uchar4* result, cudaTextureObject_t text,
                                    const int cols, const int rows,
                                    float *x_pos, float *y_pos,
                                    const float* mat3x3, bool inverse, cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(cols, blocks.x), DivUp(rows, blocks.y));
    if (inverse)
        apply_perspective_inverse<<<grid, blocks, 0, stream>>> (x_pos, y_pos, mat3x3, cols, rows);
    else
        apply_perspective<<<grid, blocks, 0, stream>>> (x_pos, y_pos, mat3x3, cols, rows);
    getLastCudaError("Apply perspective launch failed");

    resample_2D<<<grid, blocks, 0, stream>>> (result, text, cols, rows, x_pos, y_pos);
    getLastCudaError("Resample 2D image launch failed");
}


void resample_mask(unsigned char* result, cudaTextureObject_t text, const int cols,
                   const int rows, const float* x_pos, const float* y_pos,
                   const float threshold, cudaStream_t stream)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(cols, blocks.x), DivUp(rows, blocks.y));
    resample_mask_2D<<<grid, blocks, 0, stream>>>(result, text, cols, rows, x_pos, y_pos, threshold);
    getLastCudaError("Resample 2D mask launch failed");
}

void transform_blend(uchar4 * canvas, const int cw, const int ch,
                     cudaTextureObject_t frame, const int fw, const int fh,
                     const int nw, const int nh, const float* mat3x3,
                     const int tx, const int ty, cudaTextureObject_t frame_mask,
                     float * canvas_wts, cudaTextureObject_t frame_wts, cudaStream_t stream=0)
{
    dim3 blocks(16, 16);
    dim3 grid(DivUp(nw, blocks.x), DivUp(nh, blocks.y));
    transform_and_blend<<<grid, blocks, 0, stream>>>(canvas, cw, ch, frame, fw, fh,
                                                     nw, nh, mat3x3, tx, ty, frame_mask,
                                                     canvas_wts, frame_wts);
    getLastCudaError("Blend launch failed");
}
