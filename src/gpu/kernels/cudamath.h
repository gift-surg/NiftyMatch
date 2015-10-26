#ifndef __CUDA_MATH_H__
#define __CUDA_MATH_H__

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda.h>

extern "C" int DivUp(int a, int b);

extern "C" int DivDown(int a, int b);

extern "C" int AlignUp(int a, int b);

extern "C" int AlignDown(int a, int b);

template<typename TYPE>
void subtract(const TYPE * A, const TYPE * B,
              TYPE * C,
              const int width, const int height,
              cudaStream_t stream = 0);

template<typename TYPE>
void gradient(const TYPE * source,
              float2 * result,
              const int width, const int height,
              cudaStream_t stream = 0);

inline __host__ __device__ float mod_2pi_f (float x)
{
    while (x > (float)(2 * M_PI)) x -= (float) (2 * M_PI) ;
    while (x < 0.0F) x += (float) (2 * M_PI);
    return x;
}
#endif
