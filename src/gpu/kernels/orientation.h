#ifndef __ORIENTATION_H__
#define __ORIENTATION_H__

#include "cuda_runtime_api.h"

void detect_orientations(const float4 * key_pts, const float2 * grad,
                         const int num_pts,
                         const int octave_width, const int octave_height,
                         float gauss_fac, const float xper,
                         float2 * result,
                         cudaStream_t stream = 0);

#endif
