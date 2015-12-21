#include "descriptor.h"
#include "cudamath.h"
#include "helper_math.h"
#include "helper_cuda.h"

#define NBO 8
#define NBP 4

#define MAG 3
#define DESC_LENGTH 128

#define MACHINE_EPS 1.e-07

inline __device__ float normalize_histogram(float *data)
{
    float  norm = 0.0;
    const int size = NBO*NBP*NBP;
    for (int i = 0; i < size; ++i) {
        norm += data[i] * data[i];
    }

    norm = sqrt(norm) + MACHINE_EPS;

    for (int i = 0; i < size; ++i) {
        data[i] /= norm;
    }

    return norm;
}


__global__ void kernel_descriptor_optim(const float4 * keypts, const float2 * orients, const float2 * grad,
                                        const int num_pts, const int octave_width, const int octave_height,
                                        const int num_dogs, const float xper, float *desc,
                                        float *xp, float *yp)
{
    const int pt_idx = blockIdx.x;

    if (pt_idx >= num_pts) return;

    float x = keypts[pt_idx].x / xper;
    float y = keypts[pt_idx].y / xper;
    float s = keypts[pt_idx].z / xper;

    int xi = (int)(x + 0.5);
    int yi = (int)(y + 0.5);
    int si = keypts[pt_idx].w;

    if (xi < 0 || xi >= octave_width || yi < 0 || yi >= octave_height || si < 0 || si >= num_dogs)
        return;

    const float wsigma = NBP / 2;

    const float SBP = MAG * s + MACHINE_EPS;
    const int W = (int)(floor(sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5));

    const int xmin = max(-W, -xi);
    const int xmax = min(W, octave_width - 1 -xi);
    const int ymin = max(-W, -yi);
    const int ymax = min(W, octave_height -1 -yi);

    // We will need to break the computation into separate chucks.
    // We assume that blockDim.x == blockDim.y
    const int max_dims = max(xmax - xmin, ymax - ymin);
    const int chunks = ceil((max_dims + 1.f)/blockDim.x);

    const int binto = 1 ;          /* bin theta-stride */
    const int binyo = NBO * NBP ;  /* bin y-stride */
    const int binxo = NBO ;        /* bin x-stride */

    // Set the descriptor to 0
    const int index = threadIdx.y * blockDim.x + threadIdx.x;
    if (index < DESC_LENGTH) desc[pt_idx * DESC_LENGTH + index] = 0;

    if (index == 0) {
        xp[pt_idx] = keypts[pt_idx].x; yp[pt_idx] = keypts[pt_idx].y;
    }

    __syncthreads();

    float * pix_d = &desc[pt_idx * DESC_LENGTH] + (NBP/2) * binyo + (NBP/2) * binxo;

    const int grad_index = (si * octave_height + yi) * octave_width + xi;
    const float2 * grad_ptr = &grad[grad_index];

    int cx = (int)threadIdx.x + xmin;
    int cy = (int)threadIdx.y + ymin;

    const float2 angle0 = orients[pt_idx];
    double const st0 = sin (angle0.x);
    double const ct0 = cos (angle0.x);

#pragma unroll
    for (int i = 0; i < chunks; ++i)
    {
        if (cx <= xmax && cy <= ymax)
        {
            float mod = grad_ptr[cy*octave_width+cx].x;
            float ang = grad_ptr[cy*octave_width+cx].y;
            float theta = mod_2pi_f(ang-orients[pt_idx].x);

            float dx = xi + cx - x;
            float dy = yi + cy - y;
            float nx = ( ct0 * dx + st0 * dy) / SBP;
            float ny = (-st0 * dx + ct0 * dy) / SBP;

            float nt = NBO * theta / (2 * M_PI);
            float win = exp((nx*nx + ny*ny)/(2.0 * wsigma * wsigma));

            int binx = (int)floor(nx - 0.5);
            int biny = (int)floor(ny - 0.5);
            int bint = (int)floor(nt);
            float rbinx = nx - (binx + 0.5);
            float rbiny = ny - (biny + 0.5);
            float rbint = nt - bint;

#pragma unroll
            for(int dbinx = 0 ; dbinx < 2 ; ++dbinx) {
#pragma unroll
                for(int dbiny = 0 ; dbiny < 2 ; ++dbiny) {
#pragma unroll
                    for(int dbint = 0 ; dbint < 2 ; ++dbint) {
                        if (binx + dbinx >= -(NBP/2) &&
                            binx + dbinx < (NBP/2) &&
                            biny + dbiny >= -(NBP/2) &&
                            biny + dbiny <  (NBP/2))
                        {
                            float wt = win
                                    * mod
                                    * abs (1.f - dbinx - rbinx)
                                    * abs (1.f - dbiny - rbiny)
                                    * abs (1.f - dbint - rbint);

                            int loc = (binx + dbinx)*binxo + (biny + dbiny)*binyo + ((bint + dbint)*binto) % NBO;
                            atomicAdd(&pix_d[loc], wt);
                        }
                    }
                }
            }
        }

        cx += blockDim.x;
        cy += blockDim.y;
    }
}


__global__ void kernel_descriptor_naive(const float4 * kpts, const float2 * orients, const float2 * grad,
                                        const int num_pts, const int octave_width, const int octave_height,
                                        const int num_levels, const float xper, float * desc,
                                        float* xp, float* yp)
{
    int idx = blockIdx.x *blockDim.x + threadIdx.x;
    if (idx >= num_pts) return;

    float x = kpts[idx].x / xper;
    float y = kpts[idx].y / xper;
    float s = kpts[idx].z / xper;

    xp[idx] = x; yp[idx] = y;

    int xi = (int)(x + 0.5);
    int yi = (int)(y + 0.5);
    int si = kpts[idx].w;

    const float SBP = MAG * s + MACHINE_EPS;
    const float W = floor(sqrt(2.0) * SBP * (NBP + 1) / 2.0 + 0.5);

    const int binto = 1 ;          // bin theta-stride
    const int binyo = NBO * NBP ;  // bin y-stride
    const int binxo = NBO ;        // bin x-stride

    const float2 angle0 = orients[idx];
    double const st0 = sin (angle0.x) ;
    double const ct0 = cos (angle0.x) ;

    const float wsigma = NBP / 2;

    if (xi < 0 || xi >= octave_width || yi < 0 || yi >= octave_height || si < 0 || si >= num_levels)
        return;

    const int grad_index = (si * octave_height + yi) * octave_width + xi;
    const float2 * grad_ptr = &grad[grad_index];

    float * pix_d = &desc[idx * 128] + (NBP/2) * binyo + (NBP/2) * binxo;

#define atd(dbinx,dbiny,dbint) *(pix_d + (dbint)*binto + (dbiny)*binyo + (dbinx)*binxo)

    const int xmin = max(-W, 1.f-xi);
    const int xmax = min(W, octave_width-2.f-xi);
    const int ymin = max(-W, 1.f-yi);
    const int ymax = min(W, octave_height-2.f-yi);

#pragma unroll
    for(int dyi = ymin; dyi <= ymax; ++dyi) {
#pragma unroll
        for(int dxi = xmin; dxi <= xmax; ++dxi) {
            float mod = grad_ptr[dxi+dyi*octave_width].x;
            float ang = grad_ptr[dxi+dyi*octave_width].y;
            float theta = mod_2pi_f(ang-orients[idx].x);

            float dx = xi + dxi - x;
            float dy = yi + dyi - y;

            float nx = ( ct0 * dx + st0 * dy) / SBP;
            float ny = (-st0 * dx + ct0 * dy) / SBP;
            float nt = NBO * theta / (2 * M_PI);
            float win = exp((nx*nx + ny*ny)/(2.0 * wsigma * wsigma));

            int binx = (int)floor(nx - 0.5);
            int biny = (int)floor(ny - 0.5);
            int bint = (int)floor(nt);
            float rbinx = nx - (binx + 0.5);
            float rbiny = ny - (biny + 0.5);
            float rbint = nt - bint;

#pragma unroll
            for(int dbinx = 0 ; dbinx < 2 ; ++dbinx) {
#pragma unroll
                for(int dbiny = 0 ; dbiny < 2 ; ++dbiny) {
#pragma unroll
                    for(int dbint = 0 ; dbint < 2 ; ++dbint) {
                        if (binx + dbinx >= -(NBP/2) &&
                                binx + dbinx < (NBP/2) &&
                                biny + dbiny >= -(NBP/2) &&
                                biny + dbiny <  (NBP/2) )
                        {
                            float wt = win
                                    * mod
                                    * abs (1.f - dbinx - rbinx)
                                    * abs (1.f - dbiny - rbiny)
                                    * abs (1.f - dbint - rbint);

                            atd(binx+dbinx, biny+dbiny, (bint+dbint) % NBO) += wt;
                        }
                    }
                }
            }
        }
    }
}

void compute_sift_descriptors(const float4* key_pts, const float2* orients, const float2* grad, const int num_pts,
                              const int octave_width, const int octave_height, const int num_dogs,
                              const float xper, float * desc, float* x, float* y, cudaStream_t stream)
{
    {
        dim3 blocks(16, 16);
        dim3 grid(num_pts);
        kernel_descriptor_optim<<<grid, blocks, 0, stream>>>(key_pts, orients, grad, num_pts,
                                                             octave_width, octave_height, num_dogs, xper,
                                                             desc, x, y);
        getLastCudaError("SIFT descriptor detection launch failed");
    }
}
