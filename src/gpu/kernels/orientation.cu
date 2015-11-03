#include "orientation.h"
#include "cudamath.h"
#include "helper_cuda.h"
#include "helper_math.h"

#include <iostream>

#define NBINS 36


__global__ void kernel_orientations_optim(const float4* keypts, const float2* grad, const int num_pts,
                                          const int octave_width, const int octave_height,
                                          const float gaussian_factor, const float xper,
                                          float2* result)
{
    const int pt_idx = blockIdx.x;
    if (pt_idx >= num_pts || keypts[pt_idx].w < 0) return;

    float x = keypts[pt_idx].x / xper;
    float y = keypts[pt_idx].y / xper;
    float s = keypts[pt_idx].z / xper;

    int xi = (int)(x + 0.5);
    int yi = (int)(y + 0.5);

    const float sigma_w = gaussian_factor * s;
    int W = max((int)floorf(3 * sigma_w), 1);

    W = min(blockDim.x/2 - 1, W);
    W = min(blockDim.y/2 - 1, W);

    const int grad_index = ((keypts[pt_idx].w) * octave_height + yi) * octave_width + xi;
    const float2 * g = &grad[grad_index];

    __shared__ float hist[NBINS];

    // Initialize the histogram
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    if (index < NBINS) hist[index] = 0.f;

    __syncthreads();

    const int xmin = max(-W, -xi);
    const int xmax = min(W, octave_width - 1 - xi);
    const int ymin = max(-W, -yi);
    const int ymax = min(W, octave_height - 1 - yi);

    const int cx = (int)threadIdx.x + xmin;
    const int cy = (int)threadIdx.y + ymin;

    if (cx <= xmax && cy <= ymax) {
        float dx = (float)(cx + xi) -x;
        float dy = (float)(cy + yi) -y;
        float r2 = dx * dx + dy * dy;
        if (r2 < W*W+0.6) {
            float wgt = exp(r2/(2*sigma_w*sigma_w));
            int bin = (int)floorf(NBINS * g[cy*octave_width+cx].y/(2 * M_PI));
            atomicAdd(&hist[bin % NBINS], g[cy*octave_width+cx].x * wgt);
        }
    }

    __syncthreads();

    // Smooth the histogram
    __shared__ float temp[NBINS];


    if (index < NBINS - 1) {
        float prev;
#pragma unroll
        for (int iter = 0; iter < 6; ++iter) {
            if (index == 0) prev = hist[NBINS -1];
            else prev = hist[index - 1];

            float newh = (prev + hist[index] + hist[(index+1) % NBINS]) / 3.0;
            temp[index] = newh;

            if (index == 0) {
                hist[NBINS - 1] = (hist[NBINS - 2] + hist[NBINS - 1] + hist[0]) / 3.0;
            }
            __syncthreads();

            hist[index] = temp[index];
            __syncthreads();
        }
    }

    __syncthreads();


    __shared__ float maxh, threshold;
    if (index == 0) {
        maxh = 0.f;
#pragma unroll
        for (int i = 0; i < NBINS; ++i) maxh = fmaxf(maxh, hist[i]);
        threshold = maxh * 0.8;
    }

    if (index < NBINS) temp[index] = -1;

    __syncthreads();

    if (index < NBINS) {
        float h0 = hist [index] ;
        float hm = hist [(index - 1 + NBINS) % NBINS];
        float hp = hist [(index + 1 + NBINS) % NBINS];
        if (h0 > threshold && h0 > hm && h0 > hp) {
            float di = - 0.5 * (hp - hm) / (hp + hm - 2 * h0) ;
            float th = 2 * M_PI * (index + di + 0.5) / NBINS;
            temp[index] = th;
        }
    }

    __syncthreads();

    // Use one thread to collect the result
    if (index == 0) {
        int nangles = 0;
        for (int i = 0; i < NBINS; ++i) {
            if (temp[i] != -1) {
                if (nangles == 0) result[pt_idx].x = temp[i];
                else result[pt_idx].y = temp[i];
                ++nangles;

                if (nangles == 2) break;
            }
        }
    }
}


__global__ void kernel_orientations_naive(const float4* keypts, const float2* grad, const int num_pts,
                                          const int octave_width, const int octave_height,
                                          float gaussian_factor, float xper, float2* result)
{
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    if (idx >= num_pts) return;

    if (keypts[idx].w < 0) return;

    float x = keypts[idx].x / xper;
    float y = keypts[idx].y / xper;
    float s = keypts[idx].z / xper;

    int xi = (int)(x + 0.5);
    int yi = (int)(y + 0.5);

    const float sigma_w = gaussian_factor * s;
    int W = max((int)floorf(3 * sigma_w), 1);

    float hist[NBINS];

#pragma unroll
    for (int i = 0; i < NBINS; ++i) hist[i] = 0;
    const int grad_index = ((keypts[idx].w) * octave_height + yi) * octave_width + xi;

    const float2 * g = &grad[grad_index];
    const int xmin = max(-W, -xi);
    const int xmax = min(W, octave_width - 1 - xi);

    const int ymin = max(-W, -yi);
    const int ymax = min(W, octave_height - 1 - yi);

#pragma unroll
    for (int ys = ymin; ys <= ymax; ++ys) {
#pragma unroll
        for (int xs = xmin; xs <= xmax; ++xs) {
            float dx = (float)(xi + xs) - x;
            float dy = (float)(yi + ys) - y;
            float r2 = dx * dx + dy * dy;
            if (r2 >= W*W+0.6) continue;
            float wgt = exp(r2/(2*sigma_w*sigma_w));
            int bin = (int)floorf(NBINS * g[ys*octave_width+xs].y/(2 * M_PI));
            hist[bin % NBINS] += g[ys*octave_width+xs].x * wgt;
        }
    }


#pragma unroll
    // Smooth histogram
    for (int iter = 0; iter < 6; ++iter) {
        float prev  = hist[NBINS-1] ;
        float first = hist[0] ;
        int i;
#pragma unroll
        for (i = 0; i < NBINS - 1; i++) {
            float newh = (prev + hist[i] + hist[(i+1) % NBINS]) / 3.0;
            prev = hist[i] ;
            hist[i] = newh ;
        }
        hist[i] = (prev + hist[i] + first) / 3.0 ;
    }

    float maxh = 0;

#pragma unroll
    for (int i = 0; i < NBINS; ++i) maxh = fmaxf(maxh, hist[i]);

    int nangles = 0;
    const float threshold = maxh * 0.8;

#pragma unroll
    for (int i = 0; i < NBINS; ++i) {
        float h0 = hist [i] ;
        float hm = hist [(i - 1 + NBINS) % NBINS];
        float hp = hist [(i + 1 + NBINS) % NBINS];
        if (h0 > threshold && h0 > hm && h0 > hp) {
            float di = - 0.5 * (hp - hm) / (hp + hm - 2 * h0) ;
            float th = 2 * M_PI * (i + di + 0.5) / NBINS;
            if (nangles == 0) result[idx].x = th;
            else result[idx].y = th;
            ++nangles;
            if( nangles == 2) break;
        }
    }
}


void detect_orientations(const float4* key_pts, const float2* grad, const int num_pts,
                         const int octave_width, const int octave_height,
                         float gauss_factor, const float xper,
                         float2* result, cudaStream_t stream)
{
    dim3 blocks(22, 22);
    dim3 grid(num_pts);
    kernel_orientations_optim<<<grid, blocks, 0, stream>>>(key_pts, grad, num_pts,
                                                           octave_width, octave_height,
                                                           gauss_factor, xper, result);
    getLastCudaError("Orientation histogram launch failed");
}
