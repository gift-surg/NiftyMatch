#ifndef __ORIENTATION_H__
#define __ORIENTATION_H__

#include "cuda_runtime_api.h"

//!
//! \brief Detect keypoint orientations
//! \param key_pts keypoints
//! \param grad gradients
//! \param num_pts number of keypoints
//! \param octave_width
//! \param octave_height
//! \param gauss_factor
//! \param xper level of scaling (i.e. as a result of
//! downsampling)
//! \param result orientations saved here
//! \param stream
//!
void detect_orientations(const float4 * key_pts, const float2 * grad,
                         const int num_pts,
                         const int octave_width, const int octave_height,
                         float gauss_factor, const float xper,
                         float2 * result,
                         cudaStream_t stream = 0);

#endif
