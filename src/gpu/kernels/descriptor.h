#ifndef __DESCRIPTOR_H__
#define __DESCRIPTOR_H__

#include "cuda_runtime.h"

void compute_sift_descriptors(const float4 *key_pts, const float2 *orients, const float2 *grad,
                              const int num_pts, const int octave_width, const int octave_height,
                              const int num_levels, const float xper, float *desc,
                              float *x, float* y, cudaStream_t stream=0);

#endif
